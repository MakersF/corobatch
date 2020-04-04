#pragma once

#include <cassert>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>
#include <thread>

#include <corobatch/corobatch.hpp>

#define MY_FWD(...) ::std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)

namespace corobatch {

constexpr auto immediate_invoke = [](auto&& f, auto&&... args) { return MY_FWD(f)(MY_FWD(args)...); };

using ImmediateInvokeType = decltype(immediate_invoke);

class InvokeOnThread {
public:

    void join() {
        for (std::thread& thread : d_threads)
        {
            thread.join();
        }
        d_threads.clear();
    }

    ~InvokeOnThread() {
        join();
    }

    void operator()(auto&& f, auto&&... args) {
        d_threads.emplace_back(MY_FWD(f), MY_FWD(args)...);
    }

private:
    std::vector<std::thread> d_threads;
};

namespace private_ {

template<typename R, typename Arg, typename... OtherArgs>
struct VectorBatcherTypedefs
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
struct VectorBatcher : public private_::VectorBatcherTypedefs<R, Arg, OtherArgs...>
{
private:
    using Base = private_::VectorBatcherTypedefs<R, Arg, OtherArgs...>;

public:
    using StoredArgs = typename Base::StoredArgs;
    using AccumulationStorage = typename Base::AccumulationStorage;
    using ExecutedResults = typename Base::ExecutedResults;
    using Handle = typename Base::Handle;
    using Args = typename Base::Args;
    using ResultType = typename Base::ResultType;

    explicit VectorBatcher(Executor executor, F fun) : d_executor(executor), d_fun(MY_FWD(fun)) {}

    AccumulationStorage get_accumulation_storage() { return {}; }

    Handle record_arguments(AccumulationStorage& storage, Arg arg, OtherArgs... otherArgs)
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

    bool must_execute(const AccumulationStorage&) { return false; }

    void execute(AccumulationStorage&& storage, std::function<void(ExecutedResults)> callback)
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

    ResultType get_result(Handle h, const ExecutedResults& r)
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
    using VBTypedefs = private_::VectorBatcherTypedefs<R, Arg, OtherArgs...>;
    return [f = MY_FWD(fun)](typename VBTypedefs::AccumulationStorage&& storage,
                             std::function<void(typename VBTypedefs::VectorType)> callback) {
        callback(f(std::move(storage)));
    };
}

template<typename F, typename R, typename Arg, typename... OtherArgs>
using SyncFunWrapperType = decltype(get_sync_fun_wrapper<R, Arg, OtherArgs...>(std::declval<F>()));

} // namespace private_

template<typename F, typename R, typename Arg, typename... OtherArgs>
class SyncVectorBatcher
: public VectorBatcher<ImmediateInvokeType, private_::SyncFunWrapperType<F, R, Arg, OtherArgs...>, R, Arg, OtherArgs...>
{
private:
    using Base =
        VectorBatcher<ImmediateInvokeType, private_::SyncFunWrapperType<F, R, Arg, OtherArgs...>, R, Arg, OtherArgs...>;

public:
    explicit SyncVectorBatcher(F fun)
    : Base(immediate_invoke, private_::get_sync_fun_wrapper<R, Arg, OtherArgs...>(MY_FWD(fun)))
    {
    }
};

template<typename R, typename Arg, typename... OtherArgs, typename F>
auto syncVectorBatcher(F fun)
{
    return SyncVectorBatcher<F, R, Arg, OtherArgs...>(fun);
}

template<typename R, typename Arg, typename... OtherArgs, typename F, typename Executor>
auto vectorBatcher(Executor ex, F fun)
{
    return VectorBatcher<Executor, F, R, Arg, OtherArgs...>(ex, fun);
}

template<typename Batcher>
class SizedBatcher : public Batcher
{
private:
    using Base = Batcher;

public:
    using StoredArgs = typename Base::StoredArgs;
    using AccumulationStorage = std::pair<std::size_t, typename Base::AccumulationStorage>;
    using ExecutedResults = typename Base::ExecutedResults;
    using Handle = typename Base::Handle;
    using Args = typename Base::Args;
    using ResultType = typename Base::ResultType;

    explicit SizedBatcher(Batcher&& batcher, std::optional<std::size_t> maxBatchSize)
    : Batcher(MY_FWD(batcher)), d_maxBatchSize(maxBatchSize)
    {
    }

    AccumulationStorage get_accumulation_storage() { return std::make_pair(0, Base::get_accumulation_storage()); }

    template<typename... Args>
    Handle record_arguments(AccumulationStorage& storage, Args&&... args)
    {
        ++storage.first;
        return Base::record_arguments(storage.second, MY_FWD(args)...);
    }

    bool must_execute(const AccumulationStorage& storage)
    {
        return (d_maxBatchSize.has_value() and storage.first >= d_maxBatchSize.value()) or
               Base::must_execute(storage.second);
    }

    void execute(AccumulationStorage&& storage, std::function<void(ExecutedResults)> callback)
    {
        Base::execute(std::move(storage).second, std::move(callback));
    }

private:
    std::optional<std::size_t> d_maxBatchSize;
};

template<typename Batcher>
SizedBatcher(Batcher&&, std::optional<std::size_t>) -> SizedBatcher<Batcher>;

template<typename Batcher, typename WaitState>
class WaitableBatcher : public Batcher
{
private:
    using Base = Batcher;

public:
    using StoredArgs = typename Base::StoredArgs;
    using AccumulationStorage = typename Base::AccumulationStorage;
    using ExecutedResults = typename Base::ExecutedResults;
    using Handle = typename Base::Handle;
    using Args = typename Base::Args;
    using ResultType = typename Base::ResultType;

    explicit WaitableBatcher(Batcher&& batcher, WaitState&& waitState)
    : Batcher(MY_FWD(batcher)), d_waitState(MY_FWD(waitState))
    {
    }

    void execute(AccumulationStorage&& storage, std::function<void(ExecutedResults)> callback)
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

template<typename Batcher, typename WaitState>
WaitableBatcher(Batcher&& batcher, WaitState&& waitState) -> WaitableBatcher<Batcher, WaitState>;

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
