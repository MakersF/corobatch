#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include <immintrin.h>

#include <benchmark/benchmark.h>

#define COROBATCH_TRANSLATION_UNIT
#include <corobatch/corobatch.hpp>
#include <corobatch/utility/allocator.hpp>
#include <corobatch/utility/executor.hpp>

// Perform a fused multiply add operations on packs of 8 floats
class FusedMulAdd
{
    // Allow to access the pack as an array
    union FloatPack
    {
        __m256 m256;
        alignas(16) std::array<float, 8> array;

        friend std::ostream& operator<<(std::ostream& os, FloatPack s)
        {
            os << "[ " << s.array[0] << ", " << s.array[1] << ", " << s.array[2] << ", " << s.array[3] << ", "
               << s.array[4] << ", " << s.array[5] << ", " << s.array[6] << ", " << s.array[7] << " ]";
            return os;
        }
    };

    struct Storage
    {
        FloatPack a;
        FloatPack b;
        FloatPack c;
        unsigned char size;
    };

public:
    using AccumulationStorage = Storage;
    using ExecutedResults = FloatPack;
    using Handle = unsigned char;
    using Args = corobatch::ArgTypeList<float, float, float>;
    using ResultType = float;

    AccumulationStorage get_accumulation_storage() const { return {{}, {}, {}, 0}; }

    Handle record_arguments(AccumulationStorage& s, float a, float b, float c) const
    {
        unsigned char i = s.size;
        s.a.array[i] = a;
        s.b.array[i] = b;
        s.c.array[i] = c;
        s.size++;
        return i;
    }

    template<typename Callback>
    void execute(AccumulationStorage&& s, Callback cb) const
    {
        // This could be invoked when the size less than 8.
        // It is fine, we'll simply have garbage for the index that weren't set,
        // but they are not going to be accessed anyway
        FloatPack result = {};
        result.m256 = _mm256_fmadd_ps(s.a.m256, s.b.m256, s.c.m256);
        // std::cout << s.a << " * " << s.b << " + " << s.c << " = " << result;
        cb(result);
    }

    ResultType get_result(Handle h, ExecutedResults& r) const { return r.array[h]; }

    bool must_execute(const AccumulationStorage& s) const { return s.size >= 8; }
};

static auto setup_data(std::size_t size)
{
    std::default_random_engine rng{std::random_device{}()};
    std::vector<float> as(size, 0.f);
    std::iota(as.begin(), as.end(), 0);
    std::shuffle(as.begin(), as.end(), rng);
    std::vector<float> bs = as;
    std::shuffle(bs.begin(), bs.end(), rng);
    std::vector<float> cs = as;
    std::shuffle(cs.begin(), cs.end(), rng);
    return std::make_tuple(as, bs, cs);
}

template<typename task, bool use_custom_promise_allocator>
struct fma_corobatch_action
{

    static inline auto action = [](float a, float b, float c, auto& fmadd) -> task {
        float d = co_await fmadd(a, b, c);
        co_return d;
    };

    task invoke(float a, float b, float c, auto& fmadd) { return action(a, b, c, fmadd); }
};

template<typename task>
struct fma_corobatch_action<task, true>
{

    corobatch::PoolAlloc allocator;

    static inline auto action =
        [](std::allocator_arg_t, auto /* allocator */, float a, float b, float c, auto& fmadd) -> task {
        float d = co_await fmadd(a, b, c);
        co_return d;
    };

    task invoke(float a, float b, float c, auto& fmadd)
    {
        return action(std::allocator_arg, allocator.allocator<std::byte>(), a, b, c, fmadd);
    }
};

template<bool declare_callback_type,
         bool use_custom_promise_allocator,
         bool use_custom_batch_allocator,
         bool use_custom_coro_rescheduler,
         bool use_custom_executor>
float fma_corobatch_sum(const std::vector<float>& as, const std::vector<float>& bs, const std::vector<float>& cs)
{
    float sum = 0;
    auto onDone = [&sum](float result) {
        // std::cout << "result = " << result << std::endl;
        sum += result;
    };

    // At most 8 coroutines will be scheduled at the same time
    using FixedSizeExecutor = corobatch::FixedSizeExecutor<8>;
    using Executor = std::conditional_t<use_custom_executor, FixedSizeExecutor, corobatch::Executor>;
    Executor executor;

    using task_param = corobatch::task_param<float>;
    // Declare the type of the callback if the parameter is true
    using task_callback = std::
        conditional_t<declare_callback_type, typename task_param::template with_callback<decltype(onDone)>, task_param>;
    // Use the custom executor if the parameter is true
    using task_executor = std::
        conditional_t<use_custom_executor, typename task_callback::template with_executor<Executor>, task_callback>;
    using task = typename task_executor::task;

    fma_corobatch_action<task, use_custom_promise_allocator> action;

    FusedMulAdd fmaddAccumulator;
    corobatch::PoolAlloc batchallocator;
    auto fmadd = [&]() {
        if constexpr (use_custom_batch_allocator and use_custom_coro_rescheduler)
        {
            return corobatch::Batcher(std::allocator_arg,
                                      batchallocator.allocator<void>(),
                                      fmaddAccumulator,
                                      corobatch::fixedSingleExecutorRescheduler<8>(executor));
        }
        else if constexpr (use_custom_batch_allocator)
        {
            return corobatch::Batcher(std::allocator_arg, batchallocator.allocator<void>(), fmaddAccumulator);
        }
        else if constexpr (use_custom_coro_rescheduler)
        {
            return corobatch::Batcher(fmaddAccumulator, corobatch::fixedSingleExecutorRescheduler<8>(executor));
        }
        else
        {
            return corobatch::Batcher(fmaddAccumulator);
        }
    }();

    for (std::size_t i = 0; i < as.size(); i++)
    {
        corobatch::submit(executor, onDone, action.invoke(as[i], bs[i], cs[i], fmadd));
    }

    corobatch::force_execution(fmadd);
    executor.run();
    return sum;
}

static void BM_fma_loop(benchmark::State& state)
{
    const auto& [as, bs, cs] = setup_data(static_cast<std::size_t>(state.range(0)));
    for (auto _ : state)
    {
        float sum = 0;
        for (std::size_t i = 0; i < as.size(); i++)
        {
            sum += as[i] * bs[i] + cs[i];
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }
}
// Register the function as a benchmark
BENCHMARK(BM_fma_loop)->Range(8, 8 << 10);

template<bool declare_callback_type,
         bool use_custom_promise_allocator,
         bool use_custom_batch_allocator,
         bool use_custom_coro_rescheduler,
         bool use_custom_executor>
static void BM_fma_corobatch(benchmark::State& state)
{
    const auto& [as, bs, cs] = setup_data(state.range(0));
    for (auto _ : state)
    {
        float sum = fma_corobatch_sum<declare_callback_type,
                                      use_custom_promise_allocator,
                                      use_custom_batch_allocator,
                                      use_custom_coro_rescheduler,
                                      use_custom_executor>(as, bs, cs);
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }
}

struct CbType
{
    static const bool No = false;
    static const bool Yes = true;
};
struct PromPoolAlloc
{
    static const bool No = false;
    static const bool Yes = true;
};
struct BatchPoolAlloc
{
    static const bool No = false;
    static const bool Yes = true;
};
struct SingleExecResch
{
    static const bool No = false;
    static const bool Yes = true;
};
struct RingBuffExec
{
    static const bool No = false;
    static const bool Yes = true;
};

BENCHMARK_TEMPLATE(
    BM_fma_corobatch, CbType::No, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No)
    ->Range(8, 8 << 10);
BENCHMARK_TEMPLATE(
    BM_fma_corobatch, CbType::Yes, PromPoolAlloc::No, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No)
    ->Range(8, 8 << 10);
BENCHMARK_TEMPLATE(
    BM_fma_corobatch, CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::No, SingleExecResch::No, RingBuffExec::No)
    ->Range(8, 8 << 10);
BENCHMARK_TEMPLATE(
    BM_fma_corobatch, CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::No, RingBuffExec::No)
    ->Range(8, 8 << 10);
BENCHMARK_TEMPLATE(
    BM_fma_corobatch, CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::No)
    ->Range(8, 8 << 10);
BENCHMARK_TEMPLATE(
    BM_fma_corobatch, CbType::Yes, PromPoolAlloc::Yes, BatchPoolAlloc::Yes, SingleExecResch::Yes, RingBuffExec::Yes)
    ->Range(8, 8 << 10);

BENCHMARK_MAIN();
