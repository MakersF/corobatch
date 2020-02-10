#include <algorithm>
#include <cassert>
#include <deque>
#include <experimental/coroutine>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
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
        using OnReturnValueCb = std::function<void(Q)>;

        void return_value(Q val)
        {
            assert(d_cb);
            d_cb(val);
        }

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
        void return_void() {}
    };

public:
    struct promise_type : promise_return<T>
    {
        task get_return_object() { return task{*this}; }

        std::experimental::suspend_always initial_suspend() { return {}; }
        void unhandled_exception() noexcept { std::terminate(); }
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

class IBatcherWrapper
{
public:
    virtual int getNumPendingCoros() const = 0;
    virtual void executeBatch() = 0;
    virtual ~IBatcherWrapper() = default;

protected:
    IBatcherWrapper() = default;
    IBatcherWrapper(const IBatcherWrapper&) = default;
};

class Scheduler
{
public:
    void registerBatcher(IBatcherWrapper& batcher) { d_batchers.push_back(&batcher); }

    template<typename It>
    void schedule_all(It begin, It end)
    {
        d_ready_coroutines.insert(d_ready_coroutines.end(), begin, end);
    }

    void schedule_one(std::experimental::coroutine_handle<> h) { d_ready_coroutines.push_back(h); }

    std::optional<std::experimental::coroutine_handle<>> pop_next_to_resume()
    {
        if (d_ready_coroutines.empty())
        {
            auto it = std::max_element(
                d_batchers.begin(), d_batchers.end(), [](IBatcherWrapper* left, IBatcherWrapper* right) {
                    return left->getNumPendingCoros() < right->getNumPendingCoros();
                });
            if (it == d_batchers.end() or (*it)->getNumPendingCoros() == 0)
            {
                return std::nullopt;
            }
            else
            {
                (*it)->executeBatch();
            }
        }

        assert(not d_ready_coroutines.empty());
        std::experimental::coroutine_handle<> next = d_ready_coroutines.front();
        d_ready_coroutines.pop_front();
        return next;
    }

private:
    std::vector<IBatcherWrapper*> d_batchers;
    std::deque<std::experimental::coroutine_handle<>> d_ready_coroutines;
};

template<typename... Arg>
struct ArgTypeList
{
};

#if 0

struct Batcher {
    using ExecutedResults = ...;
    using Handle = ...;
    using Args = ArgTypeList<...>;
    using ResultType = ...;

    Handle record_elements(Args&&...);

    bool must_execute();

    ExecutedResults execute();

    ResultType get_result(ExecutedResults, const Handle&);
};
#endif

template<typename Batcher, typename BatcherFactory, typename... Args>
class BatcherWrapper : public IBatcherWrapper
{
private:
    struct Batch
    {
        explicit Batch(Batcher batcher) : d_batcher(MY_FWD(batcher)) {}

        void execute(Scheduler& scheduler)
        {
            d_result = d_batcher.execute();
            scheduler.schedule_all(d_waiting_coros.begin(), d_waiting_coros.end());
            d_waiting_coros.clear();
        }

        ~Batch() { assert(d_waiting_coros.empty()); }

        Batcher d_batcher;
        std::optional<typename Batcher::ExecutedResults> d_result;
        std::vector<std::experimental::coroutine_handle<>> d_waiting_coros;
    };

    // Compiler error: it fails to link when using std::shared_ptr<Batch>.
    // For now we leak it
    using SHARED_PTR_BATCH = Batch*;

    struct Handle
    {
        typename Batcher::Handle d_batcher_handle;
        SHARED_PTR_BATCH d_batch;
    };

    struct Awaitable
    {
        bool await_ready() { return d_handle.d_batch->d_result.has_value(); }

        auto await_resume()
        {
            assert(await_ready());
            return d_handle.d_batch->d_batcher.get_result(d_handle.d_batcher_handle,
                                                          d_handle.d_batch->d_result.value());
        }

        auto await_suspend(std::experimental::coroutine_handle<> h) { d_handle.d_batch->d_waiting_coros.push_back(h); }

        Handle d_handle;
    };

public:
    BatcherWrapper(Scheduler& scheduler, BatcherFactory batcherFactory)
    : d_mainExecutor(scheduler), d_batcherFactory(MY_FWD(batcherFactory)), d_current_batch(make_new_batch())
    {
    }

    Awaitable operator()(Args... args)
    {
        typename Batcher::Handle batcherHandle = d_current_batch->d_batcher.record_arguments(MY_FWD(args)...);
        Handle handle{batcherHandle, d_current_batch};
        if (d_current_batch->d_batcher.must_execute())
        {
            executeBatch();
        }
        return Awaitable{handle};
    }

private:
    Scheduler& d_mainExecutor;
    BatcherFactory d_batcherFactory;
    SHARED_PTR_BATCH d_current_batch;

    int getNumPendingCoros() const override { return d_current_batch->d_waiting_coros.size(); }

    void executeBatch() override
    {
        d_current_batch->execute(d_mainExecutor);
        d_current_batch = make_new_batch();
    }

    SHARED_PTR_BATCH make_new_batch()
    {
        // use std::make_shared when the shared_ptr bug is fixed
        return new Batch(d_batcherFactory());
    }
};

template<typename Batcher, typename BatcherFactory, typename... Args>
BatcherWrapper<Batcher, BatcherFactory, Args...>
    construct_batcherwrapper_impl(Scheduler& scheduler, BatcherFactory&& factory, ArgTypeList<Args...>)
{
    return BatcherWrapper<Batcher, BatcherFactory, Args...>(scheduler, MY_FWD(factory));
}

template<typename BatcherFactory>
auto construct_batcherwrapper(Scheduler& scheduler, BatcherFactory&& factory)
{
    using Batcher = decltype(factory());
    using Args = typename Batcher::Args;
    return construct_batcherwrapper_impl<Batcher>(scheduler, MY_FWD(factory), Args{});
}

template<typename It, typename OnDone, typename Coro, typename... BatcherFactory>
void batch(It begin, It end, OnDone onDone, Coro&& coro, BatcherFactory&&... factory)
{
    Scheduler scheduler;

    auto batcherWrappers = std::make_tuple(construct_batcherwrapper(scheduler, factory)...);

    std::apply([&](auto&... wrapper) { (scheduler.registerBatcher(wrapper), ...); }, batcherWrappers);

    for (It current = begin; current != end; ++current)
    {
        auto task = std::apply([&](auto&... wrapper) { return coro(MY_FWD(*current), wrapper...); }, batcherWrappers);
        using Task = decltype(task);
        typename Task::Handle coro_handle = std::move(task).handle();
        coro_handle.promise().set_on_return_value_cb(
            [current, &onDone](typename Task::ReturnType val) { onDone(current, MY_FWD(val)); });
        scheduler.schedule_one(coro_handle);
    }

    std::optional<std::experimental::coroutine_handle<>> next = scheduler.pop_next_to_resume();
    while (next.has_value())
    {
        next.value().resume();
        next = scheduler.pop_next_to_resume();
    }
}

template<typename F, typename R, typename Arg, typename... OtherArgs>
struct VectorBatcher
{
    using ExecutedResults = std::vector<R>;
    using Handle = std::size_t;
    using Args = ArgTypeList<Arg, OtherArgs...>;
    using ResultType = R;

    explicit VectorBatcher(F fun, std::optional<std::size_t> maxBatchSize)
    : d_fun(MY_FWD(fun)), d_maxBatchSize(maxBatchSize)
    {
    }

    Handle record_arguments(Arg arg, OtherArgs... otherArgs)
    {
        if constexpr (sizeof...(OtherArgs) == 0)
        {
            d_params.push_back(MY_FWD(arg));
        }
        else
        {
            d_params.push_back(std::make_tuple(MY_FWD(arg), MY_FWD(otherArgs)...));
        }
        return d_params.size() - 1;
    }

    bool must_execute() { return d_maxBatchSize.has_value() and d_params.size() >= d_maxBatchSize.value(); }

    ExecutedResults execute() { return d_fun(d_params); }

    ResultType get_result(Handle h, const ExecutedResults& r) { return r[h]; }

private:
    F d_fun;
    std::optional<std::size_t> d_maxBatchSize;
    std::conditional_t<sizeof...(OtherArgs) == 0, std::vector<Arg>, std::vector<std::tuple<Arg, OtherArgs...>>>
        d_params;
};

template<typename R, typename Arg, typename... OtherArgs, typename F>
auto vectorBatcher(F fun, std::optional<std::size_t> maxBatchSize = {})
{
    return [=]() { return VectorBatcher<F, R, Arg, OtherArgs...>(fun, maxBatchSize); };
}

} // namespace corobatch

#undef MY_FWD

int main()
{

    std::vector<int> data = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

    auto action = [](int v, auto&& int2dbl, auto&& dblint2str) -> corobatch::task<std::string> {
        double r = co_await int2dbl(v);
        auto s = co_await dblint2str(r, v);
        co_return s + co_await dblint2str(r, v);
    };

    auto int2dbl = corobatch::vectorBatcher<double, int>([](const std::vector<int>& params) {
        std::vector<double> res;
        for (int v : params)
        {
            double val = v + 0.5;
            res.push_back(val);
        }
        return res;
    });

    auto dblint2str =
        corobatch::vectorBatcher<std::string, double, int>([](const std::vector<std::tuple<double, int>>& params) {
            std::vector<std::string> res;
            for (auto&& [dbl, integer] : params)
            {
                std::string val = std::to_string(dbl) + "_" + std::to_string(integer);
                res.push_back(val);
            }
            return res;
        });

    auto onComputed = [](std::vector<int>::iterator it, std::string result) {
        std::cout << *it << "=" << result << " ";
    };

    corobatch::batch(data.begin(), data.end(), onComputed, action, int2dbl, dblint2str);

    std::cout << std::endl;
    return 0;
}
