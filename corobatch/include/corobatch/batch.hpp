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
        d_ready_coroutines.insert(d_ready_coroutines.end(), new_coros.begin(), new_coros.end());
        COROBATCH_LOG_DEBUG << "Coroutines scheduled for execution";
    }

    std::optional<std::experimental::coroutine_handle<>> pop_next_coro()
    {
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

template<typename OnDone, typename ReturnType>
void submit(Executor& executor, OnDone&& onDone, task<ReturnType> taskObj)
{
    typename task<ReturnType>::Handle coro_handle = std::move(taskObj).handle();
    coro_handle.promise().set_on_return_value_cb(MY_FWD(onDone));
    coro_handle.promise().bind_executor(executor);
    coro_handle.resume();
}

inline constexpr auto sink = [](auto&&...) {};

inline void submit(Executor& executor, task<void> task) { submit(executor, sink, std::move(task)); }

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

template<typename... T>
struct is_argtypelist : std::false_type
{
};

template<typename... T>
struct is_argtypelist<ArgTypeList<T...>> : std::true_type
{
};

template<typename T>
concept ConceptArgTypeList = is_argtypelist<T>::value;

template<typename T, typename U>
// 1. Use the standard library one once it's addded.
// 2. This should be = std::is_same_v<T, U>, but the compiler deduces it to <void, ...> and always fails.
// Until that is fixed, alway assume it's true
concept ConceptIsSame = true;

// This template is never called, it's only used in the concept to check that the record_arguments
// method can take the Args...
template<typename Accumulator, typename AccumulationStorage, typename... Args>
auto accumulator_record_arguments(const Accumulator& ac, AccumulationStorage& as, ArgTypeList<Args...>)
    -> decltype(ac.record_arguments(as, std::declval<Args&&>()...));

using ExecutorCoroMap = std::unordered_map<Executor*, std::vector<std::experimental::coroutine_handle<>>>;

template<typename ResultType>
auto make_callback(std::shared_ptr<void> keep_alive, ExecutorCoroMap& waiting_coros, std::optional<ResultType>& result)
{
    return [keep_alive = std::move(keep_alive), &waiting_coros, &result](ResultType results) mutable {
        assert(not waiting_coros.empty() && "Did you call the callback twice?");
        COROBATCH_LOG_DEBUG << "Batch execution completed with result = " << PrintIfPossible(results);
        result = std::move(results);
        for (auto& [executor_ptr, coroutines] : waiting_coros)
        {
            executor_ptr->schedule_all(coroutines);
            coroutines.clear();
        }
        waiting_coros.clear();
    };
}

template<typename ResultType>
using CallbackType = decltype(make_callback(std::declval<std::shared_ptr<void*>>(),
                                            std::declval<ExecutorCoroMap&>(),
                                            std::declval<std::optional<ResultType>&>()));

} // namespace private_

template<typename Acc, typename NoRefAcc = std::remove_reference_t<Acc>>
concept ConceptAccumulator = requires(const Acc& accumulator,
                                      typename NoRefAcc::AccumulationStorage accumulation_storage,
                                      typename NoRefAcc::ExecutedResults executed_result,
                                      typename NoRefAcc::Handle handle,
                                      typename NoRefAcc::Args args,
                                      typename NoRefAcc::ResultType result_type,
                                      private_::CallbackType<typename NoRefAcc::ExecutedResults> ondone_callback)
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

template<ConceptAccumulator Accumulator, typename ArgsList>
class BatcherBase;

template<ConceptAccumulator Accumulator, typename... Args>
class BatcherBase<Accumulator, ArgTypeList<Args...>> : public IBatcher
{
private:
    using NoRefAccumulator = std::remove_reference_t<Accumulator>;

    struct Batch : std::enable_shared_from_this<Batch>
    {
        Batch(Accumulator& accumulator)
        : d_accumulator(accumulator), d_storage(d_accumulator.get_accumulation_storage())
        {
            COROBATCH_LOG_DEBUG << "New batch created";
        }
        ~Batch() { assert(d_waiting_coros.empty()); }

        void execute()
        {
            COROBATCH_LOG_DEBUG << "Executing batch";
            assert(not d_waiting_coros.empty() && "Do not execute empty batches");
            // Aliasing pointer to share ownership to the batch, but without the need
            // to expose the type.
            // This allows to have the callback type be dependent only on the result type
            auto keep_alive = std::shared_ptr<void>(this->shared_from_this(), nullptr);
            d_accumulator.execute(std::move(d_storage),
                                  make_callback(std::move(keep_alive), d_waiting_coros, d_result));
        }

        const Accumulator& d_accumulator;
        typename NoRefAccumulator::AccumulationStorage d_storage;
        std::optional<typename NoRefAccumulator::ExecutedResults> d_result;
        std::unordered_map<Executor*, std::vector<std::experimental::coroutine_handle<>>> d_waiting_coros;
    };

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
            d_batch->d_waiting_coros[std::addressof(d_executor)].push_back(h);
            std::optional<std::experimental::coroutine_handle<>> next_coro = d_executor.pop_next_coro();
            if (next_coro)
            {
                COROBATCH_LOG_DEBUG << "Passing control to next coroutine";
                return next_coro.value();
            }
            else
            {
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
        Awaitable rebind_executor(Executor& executor) &&
        {
            return Awaitable{executor, MY_FWD(d_batcher_handle), d_batch};
        }

        // private:
        typename NoRefAccumulator::Handle d_batcher_handle;
        std::shared_ptr<Batch> d_batch;
    };

public:
    BatcherBase(Accumulator accumulator) : d_accumulator(MY_FWD(accumulator)), d_current_batch(make_new_batch()) {}

    ~BatcherBase()
    {
        assert(d_current_batch->d_waiting_coros.empty() &&
               "Force the execution of the batch if it has any pending coroutines");
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

    int getNumPendingCoros() const override { return d_current_batch->d_waiting_coros.size(); }

    void executeBatch() override
    {
        d_current_batch->execute();
        d_current_batch = make_new_batch();
    }

private:
    Accumulator d_accumulator;
    std::shared_ptr<Batch> d_current_batch;

    std::shared_ptr<Batch> make_new_batch() { return std::shared_ptr<Batch>(new Batch(d_accumulator)); }
};

} // namespace private_

template<ConceptAccumulator Accumulator>
class Batcher : public private_::BatcherBase<Accumulator, typename std::remove_reference_t<Accumulator>::Args>
{
private:
    using Base = private_::BatcherBase<Accumulator, typename std::remove_reference_t<Accumulator>::Args>;

public:
    using Base::Base;
};

template<ConceptAccumulator Accumulator>
Batcher(Accumulator&&) -> Batcher<Accumulator>;

template<typename... Accumulators>
auto make_batchers(Accumulators&&... accumulators)
{
    return std::make_tuple(Batcher(MY_FWD(accumulators))...);
}

} // namespace corobatch

#undef MY_FWD
#endif
