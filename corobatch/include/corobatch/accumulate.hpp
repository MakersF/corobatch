#ifndef COROBATCH_ACCUMULATE_HPP
#define COROBATCH_ACCUMULATE_HPP

#include <cassert>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include <corobatch/batch.hpp>
#include <corobatch/private_/log.hpp>

#define MY_FWD(...) ::std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)

namespace corobatch {

constexpr auto immediate_invoke = [](auto&& f, auto&&... args) { return MY_FWD(f)(MY_FWD(args)...); };

using ImmediateInvokeType = decltype(immediate_invoke);

class InvokeOnThread
{
public:
    void join()
    {
        for (std::thread& thread : d_threads)
        {
            thread.join();
        }
        d_threads.clear();
    }

    ~InvokeOnThread() { join(); }

    void operator()(auto&& f, auto&&... args) { d_threads.emplace_back(MY_FWD(f), MY_FWD(args)...); }

private:
    std::vector<std::thread> d_threads;
};

namespace private_ {

template<typename R, typename Arg, typename... OtherArgs>
struct VectorAccumulatorTypedefs
{
    // Internal
    using VectorType = std::vector<R>;

    // Public
    using StoredArgs = std::conditional_t<sizeof...(OtherArgs) == 0, Arg, std::tuple<Arg, OtherArgs...>>;
    using AccumulationStorage = std::vector<StoredArgs>;
    using ExecutedResults = std::variant<VectorType, std::exception_ptr>;
    using Handle = std::size_t;
    using Args = ArgTypeList<Arg, OtherArgs...>;
    using ResultType = R;
};

} // namespace private_

template<typename Executor, typename F, typename R, typename Arg, typename... OtherArgs>
struct VectorAccumulator : public private_::VectorAccumulatorTypedefs<R, Arg, OtherArgs...>
{
private:
    using Base = private_::VectorAccumulatorTypedefs<R, Arg, OtherArgs...>;

public:
    using StoredArgs = typename Base::StoredArgs;
    using AccumulationStorage = typename Base::AccumulationStorage;
    using ExecutedResults = typename Base::ExecutedResults;
    using Handle = typename Base::Handle;
    using Args = typename Base::Args;
    using ResultType = typename Base::ResultType;

    explicit VectorAccumulator(Executor executor, F fun) : d_executor(executor), d_fun(MY_FWD(fun)) {}

    AccumulationStorage get_accumulation_storage() const { return {}; }

    Handle record_arguments(AccumulationStorage& storage, Arg arg, OtherArgs... otherArgs) const
    {
        if constexpr (sizeof...(OtherArgs) == 0)
        {
            storage.push_back(MY_FWD(arg));
        }
        else
        {
            storage.push_back(std::make_tuple(MY_FWD(arg), MY_FWD(otherArgs)...));
        }
        return storage.size() - 1;
    }

    bool must_execute(const AccumulationStorage&) const { return false; }

    void execute(AccumulationStorage&& storage, std::function<void(ExecutedResults)> callback) const
    {
        // Asych implementation, schedule the function to be executed later
        d_executor(
            [&f = d_fun](AccumulationStorage&& storage, std::function<void(ExecutedResults)> callback) mutable {
                try
                {
                    f(std::move(storage), [callback](typename Base::VectorType r) {
                        callback(ExecutedResults{std::in_place_index<0>, std::move(r)});
                    });
                }
                catch (...)
                {
                    callback(ExecutedResults{std::in_place_index<1>, std::current_exception()});
                }
            },
            std::move(storage),
            std::move(callback));
    }

    ResultType get_result(Handle h, const ExecutedResults& r) const
    {
        if (r.index() == 0)
        {
            return std::get<0>(r)[h];
        }
        else
        {
            std::rethrow_exception(std::get<1>(r));
        }
    }

private:
    Executor d_executor;
    F d_fun;
};

namespace private_ {

template<typename R, typename Arg, typename... OtherArgs, typename F>
auto get_sync_fun_wrapper(F&& fun)
{
    using VBTypedefs = private_::VectorAccumulatorTypedefs<R, Arg, OtherArgs...>;
    return [f = MY_FWD(fun)](typename VBTypedefs::AccumulationStorage&& storage,
                             std::function<void(typename VBTypedefs::VectorType)> callback) {
        callback(f(std::move(storage)));
    };
}

template<typename F, typename R, typename Arg, typename... OtherArgs>
using SyncFunWrapperType = decltype(get_sync_fun_wrapper<R, Arg, OtherArgs...>(std::declval<F>()));

} // namespace private_

template<typename F, typename R, typename Arg, typename... OtherArgs>
class SyncVectorAccumulator
: public VectorAccumulator<ImmediateInvokeType,
                           private_::SyncFunWrapperType<F, R, Arg, OtherArgs...>,
                           R,
                           Arg,
                           OtherArgs...>
{
private:
    using Base = VectorAccumulator<ImmediateInvokeType,
                                   private_::SyncFunWrapperType<F, R, Arg, OtherArgs...>,
                                   R,
                                   Arg,
                                   OtherArgs...>;

public:
    explicit SyncVectorAccumulator(F fun)
    : Base(immediate_invoke, private_::get_sync_fun_wrapper<R, Arg, OtherArgs...>(MY_FWD(fun)))
    {
    }
};

template<typename R, typename Arg, typename... OtherArgs, typename F>
auto syncVectorAccumulator(F fun)
{
    return SyncVectorAccumulator<F, R, Arg, OtherArgs...>(fun);
}

template<typename R, typename Arg, typename... OtherArgs, typename F, typename Executor>
auto vectorAccumulator(Executor ex, F fun)
{
    return VectorAccumulator<Executor, F, R, Arg, OtherArgs...>(ex, fun);
}

template<ConceptAccumulator Accumulator>
class SizedAccumulator : public Accumulator
{
private:
    using Base = Accumulator;

public:
    using StoredArgs = typename Base::StoredArgs;
    using AccumulationStorage = std::pair<std::size_t, typename Base::AccumulationStorage>;
    using ExecutedResults = typename Base::ExecutedResults;
    using Handle = typename Base::Handle;
    using Args = typename Base::Args;
    using ResultType = typename Base::ResultType;

    explicit SizedAccumulator(Accumulator&& accumulator, std::optional<std::size_t> maxBatchSize)
    : Accumulator(MY_FWD(accumulator)), d_maxBatchSize(maxBatchSize)
    {
    }

    AccumulationStorage get_accumulation_storage() const { return std::make_pair(0, Base::get_accumulation_storage()); }

    template<typename... Args>
    Handle record_arguments(AccumulationStorage& storage, Args&&... args) const
    {
        ++storage.first;
        return Base::record_arguments(storage.second, MY_FWD(args)...);
    }

    bool must_execute(const AccumulationStorage& storage) const
    {
        return (d_maxBatchSize.has_value() and storage.first >= d_maxBatchSize.value()) or
               Base::must_execute(storage.second);
    }

    void execute(AccumulationStorage&& storage, std::function<void(ExecutedResults)> callback) const
    {
        Base::execute(std::move(storage).second, std::move(callback));
    }

private:
    std::optional<std::size_t> d_maxBatchSize;
};

namespace private_ {

template<typename T>
concept HasSize = requires(T& obj)
{
    {
        obj.size()
    }
    ->private_::ConceptIsSame<std::size_t>;
};

}; // namespace private_

// Specialization for types which already keep track of their size
template<ConceptAccumulator Accumulator>
requires private_::HasSize<typename Accumulator::AccumulationStorage> class SizedAccumulator<Accumulator>
: public Accumulator
{
private:
    using Base = Accumulator;

public:
    explicit SizedAccumulator(Accumulator&& accumulator, std::optional<std::size_t> maxBatchSize)
    : Accumulator(MY_FWD(accumulator)), d_maxBatchSize(maxBatchSize)
    {
    }

    bool must_execute(const typename Base::AccumulationStorage& storage) const
    {
        return (d_maxBatchSize.has_value() and storage.size() >= d_maxBatchSize.value()) or Base::must_execute(storage);
    }

private:
    std::optional<std::size_t> d_maxBatchSize;
};

template<ConceptAccumulator Accumulator>
SizedAccumulator(Accumulator&&, std::optional<std::size_t>) -> SizedAccumulator<Accumulator>;

template<ConceptAccumulator Accumulator, typename WaitState>
class WaitableAccumulator : public Accumulator
{
private:
    using Base = Accumulator;

public:
    using StoredArgs = typename Base::StoredArgs;
    using AccumulationStorage = typename Base::AccumulationStorage;
    using ExecutedResults = typename Base::ExecutedResults;
    using Handle = typename Base::Handle;
    using Args = typename Base::Args;
    using ResultType = typename Base::ResultType;

    explicit WaitableAccumulator(Accumulator&& accumulator, WaitState&& waitState)
    : Accumulator(MY_FWD(accumulator)), d_waitState(MY_FWD(waitState))
    {
    }

    void execute(AccumulationStorage&& storage, std::function<void(ExecutedResults)> callback) const
    {
        Base::execute(std::move(storage),
                      [cookie = d_waitState.executionStarted(), cb = std::move(callback), &waitState = d_waitState](
                          ExecutedResults res) mutable {
                          cb(std::move(res));
                          waitState.executionFinished(MY_FWD(cookie));
                      });
    }

private:
    WaitState d_waitState;
};

template<ConceptAccumulator Accumulator, typename WaitState>
WaitableAccumulator(Accumulator&& accumulator, WaitState&& waitState) -> WaitableAccumulator<Accumulator, WaitState>;

struct MTWaitState
{
    using Data = int;

    Data executionStarted()
    {
        std::unique_lock lockguard(d_mutex);
        d_numPending++;
        return {};
    }

    void executionFinished(Data)
    {
        {
            std::unique_lock lockguard(d_mutex);
            d_numPending--;
        }
        d_cv.notify_one();
    }

    void wait_for_completion()
    {
        std::unique_lock lockguard(d_mutex);
        d_cv.wait(lockguard, [&]() { return d_numPending == 0; });
    }

private:
    std::mutex d_mutex;
    std::condition_variable d_cv;
    int d_numPending = 0;
};

} // namespace corobatch

#undef MY_FWD
#endif
