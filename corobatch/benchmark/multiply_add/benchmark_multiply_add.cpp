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

class FusedMulAdd
{
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

    void execute(AccumulationStorage&& s, corobatch::private_::CallbackType<ExecutedResults> cb) const
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

struct poolalloc
{
    struct header
    {
        header* next;
        size_t size;
    };

    header* root = nullptr;
    size_t total = 0;
    size_t alloc_count = 0;

    poolalloc() = default;
    poolalloc(const poolalloc&) = delete;
    poolalloc(poolalloc&& other) : root(other.root), total(other.total), alloc_count(other.alloc_count)
    {
        other.root = nullptr;
        other.total = 0;
        other.alloc_count = 0;
    }

    ~poolalloc()
    {
        auto current = root;
        while (current)
        {
            auto next = current->next;
            std::free(current);
            current = next;
        }
        COROBATCH_LOG_INFO << "allocs " << alloc_count << " total " << total;
    }

    void* allocate(size_t align, size_t sz)
    {
        COROBATCH_LOG_DEBUG << "alloc!";
        assert(sz >= sizeof(header));
        if (root && root->size >= sz)
        {
            header* mem = root;
            root = root->next;
            mem->~header();
            return static_cast<void*>(mem);
        }
        ++alloc_count;
        total += sz;

        return std::aligned_alloc(align, sz);
    }

    void deallocate(void* p, size_t sz)
    {
        COROBATCH_LOG_DEBUG << "free!";
        assert(sz >= sizeof(header));
        auto new_entry = new (p) header;
        new_entry->size = sz;
        new_entry->next = root;
        root = new_entry;
    }

    template<typename T>
    struct Allocator
    {
        Allocator(poolalloc& poolalloc) : d_poolalloc(poolalloc) {}

        poolalloc& d_poolalloc;

        using value_type = T;

        template<typename Q>
        struct rebind
        {
            using other = Allocator<Q>;
        };

        T* allocate(std::size_t num) { return static_cast<T*>(d_poolalloc.allocate(alignof(T), sizeof(T) * num)); }

        void deallocate(T* ptr, std::size_t num) { d_poolalloc.deallocate(static_cast<void*>(ptr), sizeof(T) * num); }

        bool operator==(const Allocator& o) { return &d_poolalloc == &o.d_poolalloc; }
    };
};

static poolalloc globalpoolalloc;

template<typename T>
struct GlobalPoolAllocator : poolalloc::Allocator<T>
{
    GlobalPoolAllocator() : poolalloc::Allocator<T>(globalpoolalloc) {}

    template<typename Q>
    struct rebind
    {
        using other = GlobalPoolAllocator<Q>;
    };
};

template<bool declare_callback_type, bool use_custom_promise_allocator>
float fma_corobatch_sum(const std::vector<float>& as, const std::vector<float>& bs, const std::vector<float>& cs)
{
    float sum = 0;
    auto onDone = [&sum](float result) {
        // std::cout << "result = " << result << std::endl;
        sum += result;
    };

    using task_param = corobatch::task_param<float>;
    // Declare the type of the callback if the parameter is true
    using task_callback = std::
        conditional_t<declare_callback_type, typename task_param::template with_callback<decltype(onDone)>, task_param>;
    // Use the custom allocator for promises if the parameter is true
    using task_allocator = std::conditional_t<use_custom_promise_allocator,
                                              typename task_callback::template with_alloc<GlobalPoolAllocator<void>>,
                                              task_callback>;
    using task = typename task_allocator::task;
    auto action = [](float a, float b, float c, auto&& fmadd) -> task {
        float d = co_await fmadd(a, b, c);
        co_return d;
    };

    FusedMulAdd fmaddAccumulator;
    corobatch::Executor executor;
    auto fmadd = corobatch::Batcher<FusedMulAdd>(fmaddAccumulator);
    for (std::size_t i = 0; i < as.size(); i++)
    {
        corobatch::submit(executor, onDone, action(as[i], bs[i], cs[i], fmadd));
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

template<bool declare_callback_type, bool use_custom_promise_allocator>
static void BM_fma_corobatch(benchmark::State& state)
{
    const auto& [as, bs, cs] = setup_data(state.range(0));
    for (auto _ : state)
    {
        float sum = fma_corobatch_sum<declare_callback_type, use_custom_promise_allocator>(as, bs, cs);
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }
}
BENCHMARK_TEMPLATE(BM_fma_corobatch, false, false)->Range(8, 8 << 10);
BENCHMARK_TEMPLATE(BM_fma_corobatch, true, false)->Range(8, 8 << 10);
BENCHMARK_TEMPLATE(BM_fma_corobatch, false, true)->Range(8, 8 << 10);
BENCHMARK_TEMPLATE(BM_fma_corobatch, true, true)->Range(8, 8 << 10);

BENCHMARK_MAIN();
