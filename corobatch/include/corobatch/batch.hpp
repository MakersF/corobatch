#ifndef COROBATCH_BATCH_HPP
#define COROBATCH_BATCH_HPP

#include <algorithm>
#include <cassert>
#include <deque>
#include <experimental/coroutine>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <corobatch/logging.hpp>
#include <corobatch/private_/log.hpp>

#define MY_FWD(...) ::std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)
namespace corobatch {

class Executor
{
public:
    Executor() = default;
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    void run()
    {
        COROBATCH_LOG_DEBUG << "Running ready coroutines";
        while (not d_ready_coroutines.empty())
        {
            std::experimental::coroutine_handle<> next = d_ready_coroutines.front();
            d_ready_coroutines.pop_front();
            next.resume();
        }
    }

    ~Executor() { assert(d_ready_coroutines.empty()); }

    void schedule_all(std::span<std::experimental::coroutine_handle<>> new_coros)
    {
        COROBATCH_LOG_TRACE << "Scheduling " << new_coros.size() << " coroutines";
        d_ready_coroutines.insert(d_ready_coroutines.end(), new_coros.begin(), new_coros.end());
    }

    std::optional<std::experimental::coroutine_handle<>> pop_next_coro()
    {
        COROBATCH_LOG_TRACE << "Popping next coroutine";
        if (d_ready_coroutines.empty())
        {
            return std::nullopt;
        }
        std::experimental::coroutine_handle<> next_coro = d_ready_coroutines.front();
        d_ready_coroutines.pop_front();
        return next_coro;
    }

private:
    std::deque<std::experimental::coroutine_handle<>> d_ready_coroutines;
};

namespace private_ {

template<typename F, typename... Args>
concept Invokable = std::is_invocable_v<F, Args...>;

template<typename T>
struct FunctionCallback
{
    using type = std::function<void(T)>;
};

template<>
struct FunctionCallback<void>
{
    using type = std::function<void()>;
};

template<typename Callback>
struct promise_callback_storage
{
    void set_on_return_value_cb(Callback cb)
    {
        COROBATCH_LOG_TRACE << "Set on return callback";
        assert(not d_cb);
        d_cb.emplace(std::move(cb));
    }

protected:
    std::optional<Callback> d_cb;
};

template<typename Q, typename Callback>
struct promise_return : promise_callback_storage<Callback>
{
    void return_value(Q val)
    {
        COROBATCH_LOG_TRACE << "Returning value";
        assert(this->d_cb);
        (*(this->d_cb))(MY_FWD(val));
    }
};

template<typename Callback>
struct promise_return<void, Callback> : promise_callback_storage<Callback>
{
    void return_void()
    {
        COROBATCH_LOG_TRACE << "Returning void";
        assert(this->d_cb);
        (*(this->d_cb))();
    }
};

} // namespace private_

template<typename T,
         typename Callback = typename private_::FunctionCallback<T>::type,
         typename Allocator = std::allocator<void>,
         typename Executor = corobatch::Executor>
class task
{
public:
    struct promise_type : private_::promise_return<T, Callback>
    {
    private:
        using PromiseAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<promise_type>;
        static inline PromiseAllocator allocator;

    public:
        static void* operator new(size_t sz)
        {
            COROBATCH_LOG_TRACE << "Allocating promises: " << sz;
            return std::allocator_traits<PromiseAllocator>::allocate(allocator, sz);
        }

        static void operator delete(void* p, size_t sz)
        {
            COROBATCH_LOG_TRACE << "Deallocating promises: " << sz;
            std::allocator_traits<PromiseAllocator>::deallocate(allocator, static_cast<promise_type*>(p), sz);
        }

        task get_return_object() { return task{*this}; }

        std::experimental::suspend_always initial_suspend() { return {}; }
        void unhandled_exception() noexcept
        {
            COROBATCH_LOG_ERROR << "Unhandled exception in coroutine";
            std::terminate();
        }
        std::experimental::suspend_never final_suspend() { return {}; }

        template<typename RebindableAwaitable>
        decltype(auto) await_transform(RebindableAwaitable&& awaitable)
        {
            assert(d_executor && "The executor needs to be registered in the promise when the task is started");
            return MY_FWD(awaitable).rebind_executor(*d_executor);
        }

        void bind_executor(Executor& executor)
        {
            assert(d_executor == nullptr);
            d_executor = std::addressof(executor);
        }

    private:
        Executor* d_executor = nullptr;
    };

    task(const task&) = delete;
    task(task&& other) : d_handle(other.d_handle) { other.d_handle = nullptr; }

    ~task()
    {
        COROBATCH_LOG_TRACE << "Task terminating";
        if (d_handle)
        {
            COROBATCH_LOG_TRACE << "Destructing the associated coroutine";
            d_handle.destroy();
        }
    }

private:
    using Handle = std::experimental::coroutine_handle<promise_type>;

    Handle handle() &&
    {
        Handle other = d_handle;
        d_handle = nullptr;
        return other;
    }

    template<typename E, typename OnDone, typename R, typename C, typename A>
    friend void submit(E&, OnDone&&, task<R, C, A, E>);

    explicit task(promise_type& promise) : d_handle(Handle::from_promise(promise)) {}

    Handle d_handle;
};

template<typename Executor, typename OnDone, typename ReturnType, typename Callback, typename Allocator>
void submit(Executor& executor, OnDone&& onDone, task<ReturnType, Callback, Allocator, Executor> taskObj)
{
    COROBATCH_LOG_TRACE << "Task submitted";
    typename task<ReturnType, Callback, Allocator, Executor>::Handle coro_handle = std::move(taskObj).handle();
    coro_handle.promise().set_on_return_value_cb(MY_FWD(onDone));
    coro_handle.promise().bind_executor(executor);
    coro_handle.resume();
}

template<typename T,
         typename Callback_ = typename private_::FunctionCallback<T>::type,
         typename Allocator_ = std::allocator<void>,
         typename Executor_ = corobatch::Executor>
struct task_param
{
    using ReturnType = T;
    using Callback = Callback_;
    using Allocator = Allocator_;
    using Executor = Executor_;

    template<typename NewAllocator>
    using with_alloc = task_param<T, Callback, NewAllocator, Executor>;

    template<typename NewCallback>
    using with_callback = task_param<T, NewCallback, Allocator, Executor>;

    template<typename NewExecutor>
    using with_executor = task_param<T, Callback, Allocator, NewExecutor>;

    using task = task<ReturnType, Callback, Allocator, Executor>;
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

template<typename It> // requires iterator to IBatcher*
bool force_execution(It begin, It end)
{
    auto it = std::max_element(begin, end, [](IBatcher* left, IBatcher* right) {
        return left->getNumPendingCoros() < right->getNumPendingCoros();
    });

    if (it == end or (*it)->getNumPendingCoros() == 0)
    {
        COROBATCH_LOG_DEBUG << "No batcher has pending coros";
        return false;
    }
    else
    {
        COROBATCH_LOG_DEBUG << "Forcing execution of batcher";
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

template<typename It> // requires iterator to IBatcher*
bool has_any_pending(It begin, It end)
{
    COROBATCH_LOG_DEBUG << "Checking has_any_pending";
    return std::any_of(begin, end, [](IBatcher* batcher) { return batcher->getNumPendingCoros() > 0; });
}

template<typename... Batchers>
bool has_any_pending(Batchers&... batchers)
{
    IBatcher* batcher_ptrs[] = {std::addressof(batchers)...};
    return has_any_pending(std::begin(batcher_ptrs), std::end(batcher_ptrs));
}

template<typename... Arg>
struct ArgTypeList
{
};

namespace private_ {

template<template<class...> class T, class I>
struct is_templ_instance : std::false_type
{
};

template<template<class...> class T, class... A>
struct is_templ_instance<T, T<A...>> : std::true_type
{
};

template<template<class...> class T, class I>
constexpr bool is_templ_instance_v = is_templ_instance<T, I>::value;

template<typename T>
concept ConceptArgTypeList = is_templ_instance_v<ArgTypeList, T>;

template<typename T, typename U>
// 1. Use the standard library one once it's addded.
// 2. This should be = std::is_same_v<T, U>, but the compiler deduces it to <void, ...> and always fails.
// Until that is fixed, alway assume it's true
concept ConceptIsSame = true;

template<typename T, template<class...> class Trait>
concept TraitIsTrue = Trait<T>::value;

template<typename T>
concept Allocator = requires(T& allocator, std::size_t size)
{
    allocator.allocate(size);
};

// This template is never called, it's only used in the concept to check that the record_arguments
// method can take the Args...
template<typename Accumulator, typename AccumulationStorage, typename... Args>
auto accumulator_record_arguments(const Accumulator& ac, AccumulationStorage& as, ArgTypeList<Args...>)
    -> decltype(ac.record_arguments(as, std::declval<Args&&>()...));

template<typename ResultType, typename WaitingCoroRescheduler>
auto make_callback(std::shared_ptr<void> keep_alive,
                   WaitingCoroRescheduler& waiting_coros_rescheduler,
                   std::optional<ResultType>& result)
{
    return [keep_alive = std::move(keep_alive), &waiting_coros_rescheduler, &result](ResultType results) mutable {
        assert(not waiting_coros_rescheduler.empty() && "Did you call the callback twice?");
        COROBATCH_LOG_DEBUG << "Batch execution completed with result = " << PrintIfPossible(results);
        result = std::move(results);
        waiting_coros_rescheduler.reschedule();
        COROBATCH_LOG_TRACE << "Call to reschedule() completed";
    };
}

template<typename ResultType, typename WaitingCoroRescheduler>
using CallbackType = decltype(make_callback(std::declval<std::shared_ptr<void*>>(),
                                            std::declval<WaitingCoroRescheduler&>(),
                                            std::declval<std::optional<ResultType>&>()));

template<typename T>
concept CoroReschedulerWithoutPark = requires(T& rescheduler)
{
    T(rescheduler);
    rescheduler.reschedule();
};

template<typename T, typename Executor>
concept CoroRescheduler = CoroReschedulerWithoutPark<T>and requires(T& rescheduler,
                                                                    Executor& executor,
                                                                    std::experimental::coroutine_handle<> handle)
{
    rescheduler.park(executor, handle);
};

} // namespace private_

template<typename Acc, typename WaitingCoroRescheduler, typename NoRefAcc = std::remove_reference_t<Acc>>
concept ConceptAccumulator =
    requires(const Acc& accumulator,
             typename NoRefAcc::AccumulationStorage accumulation_storage,
             typename NoRefAcc::ExecutedResults executed_result,
             typename NoRefAcc::Handle handle,
             typename NoRefAcc::Args args,
             typename NoRefAcc::ResultType result_type,
             private_::CallbackType<typename NoRefAcc::ExecutedResults, WaitingCoroRescheduler> ondone_callback)
{
    {
        args
    }
    ->private_::ConceptArgTypeList;
    {
        accumulator.get_accumulation_storage()
    }
    ->private_::ConceptIsSame<typename NoRefAcc::AccumulationStorage>;
    {
        private_::accumulator_record_arguments(accumulator, accumulation_storage, args)
    }
    ->private_::ConceptIsSame<typename NoRefAcc::Handle>;
    {
        accumulator.execute(std::move(accumulation_storage), std::move(ondone_callback))
    }
    ->private_::ConceptIsSame<typename NoRefAcc::ExecutedResults>;
    {
        accumulator.get_result(std::move(handle), executed_result)
    }
    ->private_::ConceptIsSame<typename NoRefAcc::ResultType>;
    {
        accumulator.must_execute(std::as_const(accumulation_storage))
    }
    ->private_::ConceptIsSame<bool>;
};

// Accumulator implementation format

/*
struct Accumulator {
    using AccumulationStorage = ...;
    using ExecutedResults = ...;
    using Handle = ...;
    using Args = corobatch::ArgTypeList<...>;
    using ResultType = ...;

    AccumulationStorage get_accumulation_storage();
    Handle record_arguments(AccumulationStorage& , Args&&...);
    void execute(AccumulationStorage&&, std::function<void(ExecutedResults)>);
    ResultType get_result(Handle, ExecutedResults&);
    bool must_execute(const AccumulationStorage&);
};
*/

namespace private_ {

struct MultiExecutorRescheduler
{

    ~MultiExecutorRescheduler() { assert(d_waiting_coros.empty() && "Coroutines have not been rescheduled"); }

    void reschedule()
    {
        for (auto& [executor_ptr, coroutines] : d_waiting_coros)
        {
            executor_ptr->schedule_all(coroutines);
            coroutines.clear();
        }
        d_waiting_coros.clear();
        d_num_pending = 0;
    }

    void park(Executor& e, std::experimental::coroutine_handle<> h)
    {
        d_waiting_coros[std::addressof(e)].push_back(h);
        d_num_pending++;
    }

    std::size_t num_pending() const { return d_num_pending; }

    bool empty() const { return d_num_pending == 0; }

    std::unordered_map<Executor*, std::vector<std::experimental::coroutine_handle<>>> d_waiting_coros;
    std::size_t d_num_pending;
};

template<typename Accumulator,
         private_::Allocator Allocator,
         private_::CoroReschedulerWithoutPark WaitingCoroRescheduler,
         typename ArgsList>
requires ConceptAccumulator<Accumulator, WaitingCoroRescheduler> class BatcherBase;

template<typename Accumulator,
         private_::Allocator Allocator,
         private_::CoroReschedulerWithoutPark WaitingCoroRescheduler,
         typename... Args>
requires ConceptAccumulator<
    Accumulator,
    WaitingCoroRescheduler> class BatcherBase<Accumulator, Allocator, WaitingCoroRescheduler, ArgTypeList<Args...>>
: public IBatcher
{
private:
    using NoRefAccumulator = std::remove_reference_t<Accumulator>;

    struct Batch : std::enable_shared_from_this<Batch>
    {
        Batch(Accumulator& accumulator, WaitingCoroRescheduler waiting_coros_rescheduler)
        : d_accumulator(accumulator)
        , d_storage(d_accumulator.get_accumulation_storage())
        , d_waiting_coros_rescheduler(waiting_coros_rescheduler)
        {
            COROBATCH_LOG_DEBUG << "New batch created";
        }

        void execute()
        {
            COROBATCH_LOG_DEBUG << "Executing batch";
            assert(not d_waiting_coros_rescheduler.empty() && "Do not execute empty batches");
            // Aliasing pointer to share ownership to the batch, but without the need
            // to expose the type.
            // This allows to have the callback type be dependent only on the result type
            auto keep_alive = std::shared_ptr<void>(this->shared_from_this(), nullptr);
            d_accumulator.execute(std::move(d_storage),
                                  make_callback(std::move(keep_alive), d_waiting_coros_rescheduler, d_result));
        }

        const Accumulator& d_accumulator;
        typename NoRefAccumulator::AccumulationStorage d_storage;
        std::optional<typename NoRefAccumulator::ExecutedResults> d_result;
        WaitingCoroRescheduler d_waiting_coros_rescheduler;
    };

    template<typename Executor>
    struct Awaitable
    {
        bool await_ready() { return d_batch->d_result.has_value(); }

        decltype(auto) await_resume()
        {
            assert(await_ready());
            decltype(auto) result = d_batch->d_accumulator.get_result(d_batcher_handle, d_batch->d_result.value());
            COROBATCH_LOG_DEBUG << "Resuming coro " << private_::PrintIfPossible(result);
            return result;
        }

        std::experimental::coroutine_handle<> await_suspend(std::experimental::coroutine_handle<> h)
        {
            COROBATCH_LOG_TRACE << "Parking coroutine";
            d_batch->d_waiting_coros_rescheduler.park(d_executor, h);
            std::optional<std::experimental::coroutine_handle<>> next_coro = d_executor.pop_next_coro();
            if (next_coro)
            {
                COROBATCH_LOG_DEBUG << "Passing control to next coroutine";
                return next_coro.value();
            }
            else
            {
                COROBATCH_LOG_TRACE << "No coroutine is waiting for execution";
                return std::experimental::noop_coroutine();
            }
        }

        // private:
        Executor& d_executor;
        typename NoRefAccumulator::Handle d_batcher_handle;
        std::shared_ptr<Batch> d_batch;
    };

    struct RebindableAwaitable
    {
        template<typename Executor>
        requires private_::CoroRescheduler<WaitingCoroRescheduler, Executor> Awaitable<Executor>
            rebind_executor(Executor& executor) &&
        {
            return Awaitable<Executor>{executor, MY_FWD(d_batcher_handle), d_batch};
        }

        // private:
        typename NoRefAccumulator::Handle d_batcher_handle;
        std::shared_ptr<Batch> d_batch;
    };

public:
    template<typename A = Allocator, typename R = WaitingCoroRescheduler>
    requires private_::TraitIsTrue<A, std::is_default_constructible>and
        private_::TraitIsTrue<R, std::is_default_constructible>
        BatcherBase(Accumulator accumulator) : BatcherBase(MY_FWD(accumulator), Allocator(), WaitingCoroRescheduler())
    {
    }

    template<typename T, typename R = WaitingCoroRescheduler>
    requires private_::Allocator<T>and private_::TraitIsTrue<R, std::is_default_constructible>
        BatcherBase(Accumulator accumulator, T allocator)
    : BatcherBase(MY_FWD(accumulator), MY_FWD(allocator), WaitingCoroRescheduler())
    {
    }

    template<typename T, typename A = Allocator>
        requires(not private_::Allocator<T>) and private_::TraitIsTrue<A, std::is_default_constructible> BatcherBase(
                                                     Accumulator accumulator, T coro_scheduler)
    : BatcherBase(MY_FWD(accumulator), Allocator(), MY_FWD(coro_scheduler))
    {
    }

    BatcherBase(Accumulator accumulator, Allocator allocator, WaitingCoroRescheduler coro_scheduler)
    : d_accumulator(MY_FWD(accumulator))
    , d_allocator(MY_FWD(allocator))
    , d_original_coro_scheduler(MY_FWD(coro_scheduler))
    , d_current_batch(make_new_batch())
    {
    }

    BatcherBase(const BatcherBase& other) = delete;
    BatcherBase(BatcherBase&& other) = default;

    ~BatcherBase()
    {
        COROBATCH_LOG_TRACE << "Destructing with count " << d_current_batch.use_count();
        assert(d_current_batch.use_count() <= 1 &&
               "The batcher is being destroyed but some task is still pending waiting for a result from this batch");
    }

    RebindableAwaitable operator()(Args... args)
    {
        COROBATCH_LOG_DEBUG_BLOCK
        {
            COROBATCH_LOG_STREAM << "Recording parameter";
            ((COROBATCH_LOG_STREAM << ' ' << private_::PrintIfPossible(args)), ...);
        }
        typename NoRefAccumulator::Handle batcherHandle =
            d_accumulator.record_arguments(d_current_batch->d_storage, MY_FWD(args)...);
        RebindableAwaitable awaitable{batcherHandle, d_current_batch};
        if (d_accumulator.must_execute(d_current_batch->d_storage))
        {
            executeBatch();
        }
        return awaitable;
    }

    int getNumPendingCoros() const override { return d_current_batch->d_waiting_coros_rescheduler.num_pending(); }

    void executeBatch() override
    {
        d_current_batch->execute();
        d_current_batch = make_new_batch();
    }

private:
    Accumulator d_accumulator;
    Allocator d_allocator;
    WaitingCoroRescheduler d_original_coro_scheduler; // instantiate others copying this one
    std::shared_ptr<Batch> d_current_batch;

    std::shared_ptr<Batch> make_new_batch()
    {
        return std::allocate_shared<Batch>(d_allocator, d_accumulator, d_original_coro_scheduler);
    }
};

using default_batch_allocator = std::allocator<void>;
using default_batch_rescheduler = MultiExecutorRescheduler;
} // namespace private_

template<typename Accumulator,
         private_::Allocator Allocator = private_::default_batch_allocator,
         private_::CoroReschedulerWithoutPark WaitingCoroRescheduler = private_::default_batch_rescheduler>
requires ConceptAccumulator<Accumulator, WaitingCoroRescheduler> class Batcher
: public private_::
      BatcherBase<Accumulator, Allocator, WaitingCoroRescheduler, typename std::remove_reference_t<Accumulator>::Args>
{
private:
    using Base = private_::BatcherBase<Accumulator,
                                       Allocator,
                                       WaitingCoroRescheduler,
                                       typename std::remove_reference_t<Accumulator>::Args>;

public:
    using Base::Base;
};

template<typename Accumulator>
requires ConceptAccumulator<Accumulator, private_::default_batch_rescheduler> Batcher(Accumulator&&)
    -> Batcher<Accumulator>;

template<typename Accumulator, private_::Allocator Allocator>
requires ConceptAccumulator<Accumulator, private_::default_batch_rescheduler> Batcher(Accumulator&&, Allocator)
    -> Batcher<Accumulator, Allocator, private_::default_batch_rescheduler>;

template<typename Accumulator, typename Rescheduler>
    requires(not private_::Allocator<Rescheduler>) and
    ConceptAccumulator<Accumulator, Rescheduler> Batcher(Accumulator&&, Rescheduler)
        -> Batcher<Accumulator, private_::default_batch_allocator, Rescheduler>;

template<typename Accumulator, private_::Allocator Allocator, typename Rescheduler>
requires ConceptAccumulator<Accumulator, Rescheduler> Batcher(Accumulator&&, Allocator, Rescheduler)
    -> Batcher<Accumulator, Allocator, Rescheduler>;

template<typename... Accumulators>
auto make_batchers(Accumulators&&... accumulators)
{
    return std::make_tuple(Batcher(MY_FWD(accumulators))...);
}

} // namespace corobatch

#undef MY_FWD
#endif
