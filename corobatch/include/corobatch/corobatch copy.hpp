#pragma once

#include <algorithm>
#include <cassert>
#include <deque>
#include <experimental/coroutine>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#define MY_FWD(...) ::std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)
namespace corobatch {

template<typename T>
class task
{
private:
    template<typename Q>
    struct promise_return
    {
        void return_value(Q val)
        {
            assert(this->d_cb);
            this->d_cb(MY_FWD(val));
        }

        using OnReturnValueCb = std::function<void(T)>;
        void set_on_return_value_cb(OnReturnValueCb cb)
        {
            assert(not d_cb);
            d_cb = cb;
        }

    private:
        OnReturnValueCb d_cb;
    };

    template<>
    struct promise_return<void>
    {
        void return_void()
        {
            assert(this->d_cb);
            this->d_cb();
        }

        using OnReturnValueCb = std::function<void()>;
        void set_on_return_value_cb(OnReturnValueCb cb)
        {
            assert(not d_cb);
            d_cb = cb;
        }

    private:
        OnReturnValueCb d_cb;
    };

public:
    struct promise_type : promise_return<T>
    {
        task get_return_object() { return task{*this}; }

        std::experimental::suspend_always initial_suspend() { return {}; }
        void unhandled_exception() noexcept { assert(false && "Unhandled exception in coroutine"); }
        std::experimental::suspend_never final_suspend() { return {}; }
    };

private:
public:
    using Handle = std::experimental::coroutine_handle<promise_type>;
    using ReturnType = T;

    Handle handle() &&
    {
        Handle other = d_handle;
        d_handle = nullptr;
        return other;
    }

    task(const task&) = delete;
    task(task&& other) : d_handle(other.d_handle) { other.d_handle = nullptr; }

    ~task()
    {
        if (d_handle)
        {
            d_handle.destroy();
        }
    }

private:
    explicit task(promise_type& promise) : d_handle(Handle::from_promise(promise)) {}

    Handle d_handle;
};

class IBatcher
{
public:
    virtual int getNumPendingCoros() const = 0;
    virtual void executeBatch() = 0;
    virtual ~IBatcher() = default;

protected:
    IBatcher() = default;
    IBatcher(const IBatcher&) = default;
};

template<typename OnDone, typename ReturnType>
void submit(OnDone&& onDone, task<ReturnType> taskObj)
{
    typename task<ReturnType>::Handle coro_handle = std::move(taskObj).handle();
    coro_handle.promise().set_on_return_value_cb(MY_FWD(onDone));
    coro_handle.resume();
}

inline void submit(task<void> task)
{
    submit([]() {}, std::move(task));
}

template<typename It> // It of IBatcher*
bool force_execution(It begin, It end) // requires requires (It it) { *it -> std::convertible_to<IBatcher*>}
{
    auto it = std::max_element(begin, end, [](IBatcher* left, IBatcher* right) {
        return left->getNumPendingCoros() < right->getNumPendingCoros();
    });

    if (it == end or (*it)->getNumPendingCoros() == 0)
    {
        return false;
    }
    else
    {
        (*it)->executeBatch();
        return true;
    }
}

template<typename... Batchers>
bool force_execution(Batchers&... batchers)
{
    IBatcher* batcher_ptrs[] = {std::addressof(batchers)...};
    return force_execution(std::begin(batcher_ptrs), std::end(batcher_ptrs));
}

class BatchState
{
public:
    BatchState() = default;
    BatchState(const BatchState&) = delete;
    BatchState& operator=(const BatchState&) = delete;

    void execute()
    {
        while (not d_ready_coroutines.empty())
        {
            std::experimental::coroutine_handle<> next = d_ready_coroutines.front();
            d_ready_coroutines.pop_front();
            next.resume();
        }
    }

    ~BatchState() { assert(d_ready_coroutines.empty()); }

    template<typename It>
    void schedule_all(It begin, It end)
    {
        d_ready_coroutines.insert(d_ready_coroutines.end(), begin, end);
    }

private:
    std::deque<std::experimental::coroutine_handle<>> d_ready_coroutines;
};

template<typename... Arg>
struct ArgTypeList
{
};

#if 0
struct Batcher {
    using AccumulationStorage = ...;
    using ExecutedResults = ...;
    using Handle = ...;
    using Args = ArgTypeList<...>;
    using ResultType = ...;

    AccumulationStorage get_accumulation_storage();
    Handle record_elements(AccumulationStorage& , Args&&...);
    void execute(AccumulationStorage&&, std::function<void(ExecutedResults)>);
    ResultType get_result(ExecutedResults, const Handle&);
    bool must_execute(const AccumulationStorage&);
};
#endif

template<typename Batcher, typename... Args>
class BatcherWrapper : public IBatcher
{
private:
    using NoRefBatcher = std::remove_reference_t<Batcher>;

    struct Batch
    {
        Batch(Batcher& batcher) : d_batcher(batcher), d_storage(d_batcher.get_accumulation_storage()) {}
        ~Batch() { assert(d_waiting_coros.empty()); }

        Batcher& d_batcher;
        typename NoRefBatcher::AccumulationStorage d_storage;
        std::optional<typename NoRefBatcher::ExecutedResults> d_result;
        std::vector<std::experimental::coroutine_handle<>> d_waiting_coros;
    };

    struct Awaitable
    {
        bool await_ready() { return d_batch->d_result.has_value(); }

        auto await_resume()
        {
            assert(await_ready());
            return d_batch->d_batcher.get_result(d_batcher_handle, d_batch->d_result.value());
        }

        auto await_suspend(std::experimental::coroutine_handle<> h) { d_batch->d_waiting_coros.push_back(h); }

        // private:
        typename NoRefBatcher::Handle d_batcher_handle;
        std::shared_ptr<Batch> d_batch;
    };

public:
    BatcherWrapper(BatchState& batchState, Batcher batcher)
    : d_executor(batchState), d_batcher(MY_FWD(batcher)), d_current_batch(make_new_batch())
    {
    }

    ~BatcherWrapper()
    {
        assert(d_current_batch->d_waiting_coros.empty() &&
               "Force the execution of the batch if it has any pending coroutines");
    }

    Awaitable operator()(Args... args)
    {
        typename NoRefBatcher::Handle batcherHandle =
            d_batcher.record_arguments(d_current_batch->d_storage, MY_FWD(args)...);
        Awaitable awaitable{batcherHandle, d_current_batch};
        if (d_batcher.must_execute(d_current_batch->d_storage))
        {
            executeBatch();
        }
        return awaitable;
    }

private:
    BatchState& d_executor;
    Batcher d_batcher;
    std::shared_ptr<Batch> d_current_batch;

    int getNumPendingCoros() const override { return d_current_batch->d_waiting_coros.size(); }

    void executeBatch() override
    {
        d_batcher.execute(std::move(d_current_batch->d_storage),
                          [this, currbatch = d_current_batch](typename NoRefBatcher::ExecutedResults results) mutable {
                              assert(not currbatch->d_waiting_coros.empty() && "Did you call the callback twice?");
                              currbatch->d_result = std::move(results);
                              this->d_executor.schedule_all(currbatch->d_waiting_coros.begin(),
                                                            currbatch->d_waiting_coros.end());
                              currbatch->d_waiting_coros.clear();
                          });
        d_current_batch = make_new_batch();
    }

    std::shared_ptr<Batch> make_new_batch() { return std::shared_ptr<Batch>(new Batch(d_batcher)); }
};

template<typename Batcher, typename... Args>
BatcherWrapper<Batcher, Args...>
    construct_batcherwrapper_impl(BatchState& batchState, Batcher&& batcher, ArgTypeList<Args...>)
{
    return BatcherWrapper<Batcher, Args...>(batchState, MY_FWD(batcher));
}

template<typename Batcher>
auto construct_batcherwrapper(BatchState& batchState, Batcher&& batcher)
{
    using Args = typename std::remove_reference_t<Batcher>::Args;
    return construct_batcherwrapper_impl<Batcher>(batchState, MY_FWD(batcher), Args{});
}

template<typename... Batchers>
auto make_batcher(BatchState& batchState, Batchers&&... batchers)
{
    return std::make_tuple(construct_batcherwrapper(batchState, MY_FWD(batchers))...);
}

} // namespace corobatch
#undef MY_FWD