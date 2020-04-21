// Generated on Tue 21 Apr 15:19:44 BST 2020
// Commit: 47ec27a3208d654e0b75a999b9665ac664f94336

//////////////////////////////////////////////////////////////////////
// Start file: corobatch/logging.hpp
//////////////////////////////////////////////////////////////////////

#ifndef COROBATCH_LOGGING_HPP
#define COROBATCH_LOGGING_HPP

#include <functional>
#include <iosfwd>

namespace corobatch {

enum class LogLevel
{
    TRACE = 32,
    DEBUG = 64,
    INFO = 128,
    ERROR = 256
};

// Register logger
using LoggerCb = std::function<std::ostream*(LogLevel)>;

extern LoggerCb disabled_logger;
extern LoggerCb trace_logger;
extern LoggerCb debug_logger;
extern LoggerCb info_logger;
extern LoggerCb error_logger;

void registerLoggerCb(LoggerCb);

namespace private_ {

std::ostream* getLogStream(LogLevel);

}

} // namespace corobatch

#endif

//////////////////////////////////////////////////////////////////////
// End file: corobatch/logging.hpp
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Start file: corobatch/private_/logging.cpp
//////////////////////////////////////////////////////////////////////

#ifndef COROBATCH_PRIVATE_LOGGING_CPP
#define COROBATCH_PRIVATE_LOGGING_CPP

#ifdef COROBATCH_TRANSLATION_UNIT

#include <iostream>

// #include <corobatch/logging.hpp> // Removed during single header generation

namespace corobatch {

LoggerCb disabled_logger = [](LogLevel) -> std::ostream* { return nullptr; };

LoggerCb trace_logger = [](LogLevel level) -> std::ostream* {
    if (level >= LogLevel::TRACE)
    {
        return &std::cerr;
    }
    return nullptr;
};

LoggerCb debug_logger = [](LogLevel level) -> std::ostream* {
    if (level >= LogLevel::DEBUG)
    {
        return &std::cerr;
    }
    return nullptr;
};

LoggerCb info_logger = [](LogLevel level) -> std::ostream* {
    if (level >= LogLevel::INFO)
    {
        return &std::cerr;
    }
    return nullptr;
};

LoggerCb error_logger = [](LogLevel level) -> std::ostream* {
    if (level >= LogLevel::ERROR)
    {
        return &std::cerr;
    }
    return nullptr;
};

namespace {

LoggerCb logger_cb = error_logger;

}

void registerLoggerCb(LoggerCb cb) { logger_cb = cb; }

namespace private_ {

std::ostream* getLogStream(LogLevel level) { return logger_cb(level); }

} // namespace private_

} // namespace corobatch

#endif
#endif

//////////////////////////////////////////////////////////////////////
// End file: corobatch/private_/logging.cpp
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Start file: corobatch/private_/log.hpp
//////////////////////////////////////////////////////////////////////

#ifndef COROBATCH_PRIVATE_LOG_HPP
#define COROBATCH_PRIVATE_LOG_HPP

// #include <corobatch/logging.hpp> // Removed during single header generation
#include <iomanip>
#include <iostream>

#define COROBATCH_LOGLEVEL_NAME loglevel##__LINE__
#define COROBATCH_STREAM_NAME stream##__LINE__

#define COROBATCH_PRINT_LOGLINE_PREFIX(stream) \
    stream << __FILE__ << ":" << __LINE__ << " " << (COROBATCH_LOGLEVEL_NAME) << " "

#ifndef COROBATCH_DISABLE_LOGGING
// Get the log stream ptr, check if it's valid, if it is, the body of the loop will be executed.
// In the increment statement, use the comma operator to put a new line at the end of the stream
// and then reset the pointer so that we don't enter the loop anymore
#define COROBATCH_LOG_BLOCK(level, levelname)                                                                    \
    for (const char* COROBATCH_LOGLEVEL_NAME = levelname; COROBATCH_LOGLEVEL_NAME != nullptr;                    \
         COROBATCH_LOGLEVEL_NAME = nullptr)                                                                      \
        for (::std::ostream* COROBATCH_STREAM_NAME = ::corobatch::private_::getLogStream((level));               \
             COROBATCH_STREAM_NAME != nullptr && (COROBATCH_PRINT_LOGLINE_PREFIX(*COROBATCH_STREAM_NAME), true); \
             COROBATCH_STREAM_NAME = (*(COROBATCH_STREAM_NAME) << '\n', nullptr))
#else
#define COROBATCH_LOG_BLOCK(level, levelname)                                                                       \
    for (const char* COROBATCH_LOGLEVEL_NAME = nullptr; COROBATCH_LOGLEVEL_NAME; COROBATCH_LOGLEVEL_NAME = nullptr) \
        for (::std::ostream* COROBATCH_STREAM_NAME = nullptr; COROBATCH_STREAM_NAME; COROBATCH_STREAM_NAME = nullptr)
#endif

#define COROBATCH_LOG_STREAM (*(COROBATCH_STREAM_NAME))

// Block macros

#define COROBATCH_LOG_TRACE_BLOCK COROBATCH_LOG_BLOCK(::corobatch::LogLevel::TRACE, "TRACE")

#define COROBATCH_LOG_DEBUG_BLOCK COROBATCH_LOG_BLOCK(::corobatch::LogLevel::DEBUG, "DEBUG")

#define COROBATCH_LOG_INFO_BLOCK COROBATCH_LOG_BLOCK(::corobatch::LogLevel::INFO, "INFO")

#define COROBATCH_LOG_ERROR_BLOCK COROBATCH_LOG_BLOCK(::corobatch::LogLevel::ERROR, "ERROR")

// Line macros

#define COROBATCH_LOG_TRACE COROBATCH_LOG_TRACE_BLOCK COROBATCH_LOG_STREAM
#define COROBATCH_LOG_DEBUG COROBATCH_LOG_DEBUG_BLOCK COROBATCH_LOG_STREAM
#define COROBATCH_LOG_INFO COROBATCH_LOG_INFO_BLOCK COROBATCH_LOG_STREAM
#define COROBATCH_LOG_ERROR COROBATCH_LOG_ERROR_BLOCK COROBATCH_LOG_STREAM

namespace corobatch {
namespace private_ {

template<typename T>
struct PrintIfPossible
{
    PrintIfPossible(const T& value) : d_value(value) {}

    friend std::ostream& operator<<(std::ostream& os, PrintIfPossible obj)
    {
        if constexpr (requires { os << d_value; })
        {
            os << obj.d_value;
        }
        else
        {
            // Print as bytes
            const unsigned char* begin = reinterpret_cast<const unsigned char*>(&obj.d_value);
            const unsigned char* end = begin + sizeof(obj.d_value);
            std::ios_base::fmtflags previous_flags = os.flags();
            os << "[";
            for (; begin != end; begin++)
            {
                os << " 0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(*begin);
            }
            os << " ]";
            os.flags(previous_flags);
        }
        return os;
    }

    const T& d_value;
};

} // namespace private_
} // namespace corobatch

#endif

//////////////////////////////////////////////////////////////////////
// End file: corobatch/private_/log.hpp
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Start file: corobatch/batch.hpp
//////////////////////////////////////////////////////////////////////

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

// #include <corobatch/logging.hpp> // Removed during single header generation
// #include <corobatch/private_/log.hpp> // Removed during single header generation

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

template<typename T, typename Callback, typename Executor>
struct promise_methods : promise_return<T, Callback>
{
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

template<typename Allocator>
struct promise_allocation
{
    using ByteAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<std::byte>;

    template<typename... Args>
    static void* operator new(std::size_t sz, std::allocator_arg_t, Allocator& allocator, Args&... args)
    {
        COROBATCH_LOG_TRACE_BLOCK
        {
            COROBATCH_LOG_STREAM << "Allocating new promise with custom allocator " << PrintIfPossible(allocator)
                                 << " and args";
            ((COROBATCH_LOG_STREAM << ' ' << PrintIfPossible(args)), ...);
        }
        // We allocate with the byte allocator, but we copy the original allocator in the memory
        ByteAllocator byteAllocator(allocator);

        // Round up sz to next multiple of Allocator alignment
        std::size_t allocatorOffset = (sz + alignof(Allocator) - 1u) & ~(alignof(Allocator) - 1u);
        // Call onto allocator to allocate space for coroutine frame.
        void* ptr = byteAllocator.allocate(allocatorOffset + sizeof(Allocator));

        // Take a copy of the allocator (assuming noexcept copy constructor here)
        new (((char*) ptr) + allocatorOffset) Allocator(allocator);

        return ptr;
    }

    static void operator delete(void* ptr, std::size_t sz)
    {
        std::size_t allocatorOffset = (sz + alignof(Allocator) - 1u) & ~(alignof(Allocator) - 1u);
        Allocator& allocator = *reinterpret_cast<Allocator*>(((char*) ptr) + allocatorOffset);
        COROBATCH_LOG_TRACE << "Deallocating new promise with custom allocator " << PrintIfPossible(allocator);

        // Construct the byte allocator by moving the original allocator first so it isn't freeing its
        // own memory from underneath itself.
        // Assuming allocator move-constructor is noexcept here.
        ByteAllocator byteAllocator(std::move(allocator));

        // But don't forget to destruct allocator object in coroutine frame
        allocator.~Allocator();

        // Finally, free the memory using the allocator.
        byteAllocator.deallocate(static_cast<std::byte*>(ptr), allocatorOffset + sizeof(Allocator));
    }
};

template<>
struct promise_allocation<void>
{
};

} // namespace private_

template<typename T,
         typename Callback = typename private_::FunctionCallback<T>::type,
         typename Executor = corobatch::Executor>
class task
{
public:
    task() = delete;
    task(const task&) = delete;
    task(task&& other) : d_handle(other.d_handle), d_promise_ptr(other.d_promise_ptr)
    {
        other.d_handle = nullptr;
        other.d_promise_ptr = nullptr;
    }

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
    template<typename Allocator>
    struct promise
    : private_::promise_methods<T, Callback, Executor>
    , private_::promise_allocation<Allocator>
    {
        task get_return_object()
        {
            return task(std::experimental::coroutine_handle<promise>::from_promise(*this), this);
        }
    };

    std::experimental::coroutine_handle<> handle() &&
    {
        std::experimental::coroutine_handle<> other = d_handle;
        d_handle = nullptr;
        return other;
    }

    template<typename E, typename OnDone, typename R, typename C>
    friend void submit(E&, OnDone&&, task<R, C, E>);

    template<typename, typename...>
    friend class std::experimental::coroutine_traits;

    explicit task(std::experimental::coroutine_handle<> handle,
                  private_::promise_methods<T, Callback, Executor>* promise_ptr)
    : d_handle(handle), d_promise_ptr(promise_ptr)
    {
    }

    std::experimental::coroutine_handle<> d_handle;
    private_::promise_methods<T, Callback, Executor>* d_promise_ptr;
};

} // namespace corobatch

namespace std::experimental {
template<typename... TaskArgs, typename... Args>
struct coroutine_traits<corobatch::task<TaskArgs...>, Args...>
{
    using promise_type = typename corobatch::task<TaskArgs...>::template promise<void>;
};

// Specialize for free functions
template<typename... TaskArgs, typename Allocator, typename... Args>
struct coroutine_traits<corobatch::task<TaskArgs...>, std::allocator_arg_t, Allocator, Args...>
{
    using promise_type = typename corobatch::task<TaskArgs...>::template promise<Allocator>;
};

// Specialize for member functions (and lambdas)
template<typename... TaskArgs, typename Class, typename Allocator, typename... Args>
struct coroutine_traits<corobatch::task<TaskArgs...>, Class, std::allocator_arg_t, Allocator, Args...>
{
    using promise_type = typename corobatch::task<TaskArgs...>::template promise<Allocator>;
};

} // namespace std::experimental

namespace corobatch {

template<typename Executor, typename OnDone, typename ReturnType, typename Callback>
void submit(Executor& executor, OnDone&& onDone, task<ReturnType, Callback, Executor> taskObj)
{
    COROBATCH_LOG_TRACE << "Task submitted";
    taskObj.d_promise_ptr->set_on_return_value_cb(MY_FWD(onDone));
    taskObj.d_promise_ptr->bind_executor(executor);
    std::move(taskObj).handle().resume();
}

template<typename T,
         typename Callback_ = typename private_::FunctionCallback<T>::type,
         typename Executor_ = corobatch::Executor>
struct task_param
{
    using ReturnType = T;
    using Callback = Callback_;
    using Executor = Executor_;

    template<typename NewCallback>
    using with_callback = task_param<T, NewCallback, Executor>;

    template<typename NewExecutor>
    using with_executor = task_param<T, Callback, NewExecutor>;

    using task = task<ReturnType, Callback, Executor>;
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

// These template is never called, it's only used in the concept to check that the record_arguments
// method can take the Args...
template<typename Accumulator, typename AccumulationStorage, typename... Args>
auto accumulator_record_arguments(const Accumulator& ac, AccumulationStorage& as, ArgTypeList<Args...>)
    -> decltype(ac.record_arguments(as, std::declval<Args&&>()...));

template<typename Accumulator, typename AccumulationStorage, typename... Args>
auto accumulator_must_execute_with_args(const Accumulator& ac, const AccumulationStorage& as, ArgTypeList<Args...>)
    -> decltype(ac.must_execute(as, std::declval<const Args&>()...));

template<typename Accumulator, typename AccumulationStorage, typename ArgTypeList>
concept HasMustExecuteWithArgs = requires(const Accumulator& accumulator,
                                          AccumulationStorage accumulation_storage,
                                          ArgTypeList args)
{
    {
        accumulator_must_execute_with_args(accumulator, accumulation_storage, args)
    }
    ->ConceptIsSame<bool>;
};

template<typename Accumulator, typename AccumulationStorage>
concept HasMustExecuteWithoutArgs = requires(const Accumulator& accumulator, AccumulationStorage accumulation_storage)
{
    {
        accumulator.must_execute(std::as_const(accumulation_storage))
    }
    ->private_::ConceptIsSame<bool>;
};

template<typename Accumulator, typename AccumulationStorage, typename ArgTypeList>
concept HasMustExecute = HasMustExecuteWithoutArgs<Accumulator, AccumulationStorage> or
                         HasMustExecuteWithArgs<Accumulator, AccumulationStorage, ArgTypeList>;

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
    private_::HasMustExecute<Acc, typename NoRefAcc::AccumulationStorage, typename NoRefAcc::Args>and requires(
        const Acc& accumulator,
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
        BatcherBase(Accumulator accumulator)
    : BatcherBase(std::allocator_arg, Allocator(), MY_FWD(accumulator), WaitingCoroRescheduler())
    {
    }

    template<typename T, typename R = WaitingCoroRescheduler>
    requires private_::Allocator<T>and private_::TraitIsTrue<R, std::is_default_constructible>
        BatcherBase(std::allocator_arg_t, T allocator, Accumulator accumulator)
    : BatcherBase(std::allocator_arg, MY_FWD(allocator), MY_FWD(accumulator), WaitingCoroRescheduler())
    {
    }

    template<typename T, typename A = Allocator>
        requires(not private_::Allocator<T>) and private_::TraitIsTrue<A, std::is_default_constructible> BatcherBase(
                                                     Accumulator accumulator, T coro_scheduler)
    : BatcherBase(std::allocator_arg, Allocator(), MY_FWD(accumulator), MY_FWD(coro_scheduler))
    {
    }

    BatcherBase(std::allocator_arg_t,
                Allocator allocator,
                Accumulator accumulator,
                WaitingCoroRescheduler coro_scheduler)
    : d_allocator(MY_FWD(allocator))
    , d_accumulator(MY_FWD(accumulator))
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
        if constexpr (private_::HasMustExecuteWithoutArgs<Accumulator, typename NoRefAccumulator::AccumulationStorage>)
        {
            typename NoRefAccumulator::Handle batcherHandle =
                d_accumulator.record_arguments(d_current_batch->d_storage, MY_FWD(args)...);
            RebindableAwaitable awaitable{batcherHandle, d_current_batch};
            if (d_accumulator.must_execute(std::as_const(d_current_batch->d_storage)))
            {
                executeBatch();
            }
            return awaitable;
        }
        else
        {
            if (d_accumulator.must_execute(std::as_const(d_current_batch->d_storage), std::as_const(args)...))
            {
                executeBatch();
            }
            typename NoRefAccumulator::Handle batcherHandle =
                d_accumulator.record_arguments(d_current_batch->d_storage, MY_FWD(args)...);
            RebindableAwaitable awaitable{batcherHandle, d_current_batch};
            return awaitable;
        }
    }

    int getNumPendingCoros() const override { return d_current_batch->d_waiting_coros_rescheduler.num_pending(); }

    void executeBatch() override
    {
        d_current_batch->execute();
        d_current_batch = make_new_batch();
    }

private:
    Allocator d_allocator;
    Accumulator d_accumulator;
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
requires ConceptAccumulator<Accumulator, private_::default_batch_rescheduler>
    Batcher(std::allocator_arg_t, Allocator, Accumulator&&)
        -> Batcher<Accumulator, Allocator, private_::default_batch_rescheduler>;

template<typename Accumulator, typename Rescheduler>
    requires(not private_::Allocator<Rescheduler>) and
    ConceptAccumulator<Accumulator, Rescheduler> Batcher(Accumulator&&, Rescheduler)
        -> Batcher<Accumulator, private_::default_batch_allocator, Rescheduler>;

template<typename Accumulator, private_::Allocator Allocator, typename Rescheduler>
requires ConceptAccumulator<Accumulator, Rescheduler>
    Batcher(std::allocator_arg_t, Allocator, Accumulator&&, Rescheduler)
        -> Batcher<Accumulator, Allocator, Rescheduler>;

template<typename... Accumulators>
auto make_batchers(Accumulators&&... accumulators)
{
    return std::make_tuple(Batcher(MY_FWD(accumulators))...);
}

} // namespace corobatch

#undef MY_FWD
#endif

//////////////////////////////////////////////////////////////////////
// End file: corobatch/batch.hpp
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Start file: corobatch/accumulate.hpp
//////////////////////////////////////////////////////////////////////

#ifndef COROBATCH_ACCUMULATE_HPP
#define COROBATCH_ACCUMULATE_HPP

#include <cassert>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

// #include <corobatch/batch.hpp> // Removed during single header generation
// #include <corobatch/private_/log.hpp> // Removed during single header generation

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
    // Public
    using StoredArgs = std::conditional_t<sizeof...(OtherArgs) == 0, Arg, std::tuple<Arg, OtherArgs...>>;
    using AccumulationStorage = std::vector<StoredArgs>;
    using ExecutedResults = std::vector<R>;
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

    template<private_::Invokable<ExecutedResults> Callback>
    void execute(AccumulationStorage&& storage, Callback callback) const
    {
        // Async implementation, schedule the function to be executed later
        d_executor(d_fun, std::move(storage), std::move(callback));
    }

    ResultType get_result(Handle h, const ExecutedResults& r) const { return r[h]; }

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
                             private_::Invokable<typename VBTypedefs::ExecutedResults> auto callback) {
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

template<typename Accumulator>
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

    template<private_::Invokable<ExecutedResults> Callback>
    void execute(AccumulationStorage&& storage, Callback callback) const
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
template<typename Accumulator>
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

template<typename Accumulator>
SizedAccumulator(Accumulator&&, std::optional<std::size_t>) -> SizedAccumulator<Accumulator>;

template<typename Accumulator, typename WaitState>
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

    template<private_::Invokable<ExecutedResults> Callback>
    void execute(AccumulationStorage&& storage, Callback callback) const
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

template<typename Accumulator, typename WaitState>
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
        d_cv.notify_all();
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

//////////////////////////////////////////////////////////////////////
// End file: corobatch/accumulate.hpp
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Start file: corobatch/utility/allocator.hpp
//////////////////////////////////////////////////////////////////////

#ifndef COROBATCH_UTILITY_ALLOCATOR_HPP
#define COROBATCH_UTILITY_ALLOCATOR_HPP

#include <cassert>
// #include <corobatch/private_/log.hpp> // Removed during single header generation
#include <memory>

namespace corobatch {

namespace private_ {

template<typename UnderlyingAlloc, typename T>
class AllocatorWrapper
{
private:
    UnderlyingAlloc& d_underlyingAlloc;

    template<typename, typename>
    friend class AllocatorWrapper;

public:
    explicit AllocatorWrapper(UnderlyingAlloc& underlyingAlloc) : d_underlyingAlloc(underlyingAlloc) {}

    template<typename Q>
    explicit AllocatorWrapper(const AllocatorWrapper<UnderlyingAlloc, Q> o) : AllocatorWrapper(o.d_underlyingAlloc)
    {
    }

    template<typename Q>
    struct rebind
    {
        using other = AllocatorWrapper<UnderlyingAlloc, Q>;
    };

    using value_type = T;

    T* allocate(std::size_t num) { return static_cast<T*>(d_underlyingAlloc.allocate(alignof(T), sizeof(T) * num)); }

    void deallocate(T* ptr, std::size_t num) { d_underlyingAlloc.deallocate(static_cast<void*>(ptr), sizeof(T) * num); }

    bool operator==(const AllocatorWrapper& o) { return &d_underlyingAlloc == &o.d_underlyingAlloc; }
};

} // namespace private_

// Can be used to allocate only one size and one alignment.
// Reuse deallocated blocks for new allocations;
class PoolAlloc
{
private:
    struct header
    {
        header* d_next;
    };

    header* d_root = nullptr;
    size_t d_supported_size = 0;
    size_t d_supported_alignment = 0;
    size_t d_allocations_count = 0;

public:
    PoolAlloc() = default;
    PoolAlloc(const PoolAlloc&) = delete;
    PoolAlloc(PoolAlloc&& other)
    : d_root(other.d_root)
    , d_supported_size(other.d_supported_size)
    , d_supported_alignment(other.d_supported_alignment)
    , d_allocations_count(other.d_allocations_count)
    {
        other.d_root = nullptr;
        other.d_supported_size = 0;
        other.d_supported_alignment = 0;
        other.d_allocations_count = 0;
    }

    ~PoolAlloc()
    {
        std::size_t available_blocks = 0;
        auto current = d_root;
        while (current)
        {
            auto next = current->d_next;
            std::free(current);
            available_blocks++;
            current = next;
        }
        COROBATCH_LOG_INFO << "Total allocations: " << d_allocations_count << ". Supported size: " << d_supported_size
                           << ". Supported alignment: " << d_supported_alignment;
        assert(available_blocks == d_allocations_count &&
               "The allocator has been destructed and some memory allocated through it was not freed yet");
    }

    void* allocate(std::size_t align, std::size_t sz)
    {
        COROBATCH_LOG_TRACE << "Allocating " << sz << " with alignment " << align;
        assert(sz >= sizeof(header) && "The allocated requires the size to be bigger");
        if (d_supported_size == 0)
        {
            d_supported_size = sz;
            d_supported_alignment = align;
        }
        assert(d_supported_size == sz &&
               "The allocator supports allocating only a single size. Asked a size different from a previous one");
        assert(d_supported_alignment == align && "The allocator supports allocating only with a single alignment. "
                                                 "Asked an alignment differnt from a previous one");
        if (d_root)
        {
            header* mem = d_root;
            d_root = d_root->d_next;
            mem->~header();
            return static_cast<void*>(mem);
        }
        ++d_allocations_count;
        return std::aligned_alloc(align, sz);
    }

    void deallocate(void* p, std::size_t sz)
    {
        COROBATCH_LOG_TRACE << "Deallocating " << sz;
        assert(sz >= sizeof(header));
        auto new_entry = new (p) header;
        new_entry->d_next = d_root;
        d_root = new_entry;
    }

    template<typename T>
    using Allocator = private_::AllocatorWrapper<PoolAlloc, T>;

    template<typename T>
    Allocator<T> allocator()
    {
        return Allocator<T>(*this);
    }
};

// Can be used to allocate only one size and one alignment.
// On the first allocation it allocates the memory for the expected number of items
class UniformSizeLazyBulkAlloc
{
private:
    char* d_memory = nullptr;
    std::size_t d_capacity = 0; // How much memory is allocated
    std::size_t d_supported_size = 0; // the size of the items this allocator is used to allocate
    std::size_t d_currentoffset = 0; // where the free memory begins
    std::size_t d_num_allocs = 0; // The number of active allocations
    std::size_t d_max_num_items = 0; // How many items at most this allocator can contain
    std::size_t d_max_num_allocs = 0; // The maximum number of items allocated on this allocator at the same time
public:
    UniformSizeLazyBulkAlloc(std::size_t max_num_items) : d_max_num_items(max_num_items) {}
    UniformSizeLazyBulkAlloc(const UniformSizeLazyBulkAlloc&) = delete;
    UniformSizeLazyBulkAlloc(UniformSizeLazyBulkAlloc&& other)
    : d_memory(other.d_memory)
    , d_capacity(other.d_capacity)
    , d_supported_size(other.d_supported_size)
    , d_currentoffset(other.d_currentoffset)
    , d_num_allocs(other.d_num_allocs)
    , d_max_num_items(other.d_max_num_items)
    {
        other.d_memory = nullptr;
        other.d_capacity = 0;
        other.d_supported_size = 0;
        other.d_currentoffset = 0;
        other.d_num_allocs = 0;
        other.d_max_num_items = 0;
    }

    ~UniformSizeLazyBulkAlloc()
    {
        assert(d_num_allocs == 0);
        std::free(d_memory);
        COROBATCH_LOG_INFO << "Total allocations: " << d_max_num_allocs << " out of " << d_max_num_items
                           << "(max) . Supported size: " << d_supported_size
                           << ". Total memory allocated: " << d_capacity;
    }

    void* allocate(std::size_t align, std::size_t sz)
    {
        COROBATCH_LOG_TRACE << "Allocating " << sz << " with alignment " << align;
        if (d_memory == nullptr)
        {
            d_capacity = sz * d_max_num_items;
            COROBATCH_LOG_TRACE << "Allocating " << d_capacity << " for " << d_max_num_items << " items of size " << sz;
            d_memory = static_cast<char*>(std::aligned_alloc(align, d_capacity));
            d_supported_size = sz;
        }
        assert(sz == d_supported_size && "The allocator can only allocate elements with the same size");
        assert(d_currentoffset + sz <= d_capacity && "Allocated more memory than the allocator can support");
        void* ptr = d_memory + d_currentoffset;
        std::size_t left = d_capacity - d_currentoffset;
        ptr = std::align(align, sz, ptr, left);
        assert(ptr != nullptr);
        d_currentoffset = (d_capacity - left) + sz;
        d_num_allocs++;
        d_max_num_allocs = std::max(d_max_num_allocs, d_num_allocs);
        return ptr;
    }

    void deallocate(void*, std::size_t sz)
    {
        COROBATCH_LOG_TRACE << "Deallocating " << sz;
        d_num_allocs--;
        if (d_num_allocs == 0)
        {
            d_currentoffset = 0;
        }
    }

    template<typename T>
    using Allocator = private_::AllocatorWrapper<UniformSizeLazyBulkAlloc, T>;

    template<typename T>
    Allocator<T> allocator()
    {
        return Allocator<T>(*this);
    }
};

} // namespace corobatch

#endif

//////////////////////////////////////////////////////////////////////
// End file: corobatch/utility/allocator.hpp
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Start file: corobatch/utility/executor.hpp
//////////////////////////////////////////////////////////////////////

#ifndef COROBATCH_UTILITY_EXECUTOR_HPP
#define COROBATCH_UTILITY_EXECUTOR_HPP

#include <array>
#include <cassert>
// #include <corobatch/private_/log.hpp> // Removed during single header generation

namespace corobatch {

template<std::size_t MaxConcurrentTasksWaitingOnBatch>
class StaticVector
{
public:
    auto begin() { return d_storage.begin(); }

    auto end() { return d_storage.begin() + d_size; }

    void push_back(const std::experimental::coroutine_handle<>& h)
    {
        d_storage[d_size] = h;
        d_size++;
    }

    std::experimental::coroutine_handle<>* data() { return d_storage.data(); }
    void clear() { d_size = 0; }
    bool empty() const { return d_size == 0; }
    std::size_t size() const { return d_size; }

private:
    std::array<std::experimental::coroutine_handle<>, MaxConcurrentTasksWaitingOnBatch> d_storage;
    std::size_t d_size = 0;
};

// Can be used when all the tasks are scheduled on the same executor
template<typename Executor, typename Storage = std::vector<std::experimental::coroutine_handle<>>>
class SingleExecutorRescheduler
{
public:
    SingleExecutorRescheduler() = default;
    SingleExecutorRescheduler(Executor& e) : d_executor_ptr(&e) {}

    void reschedule()
    {
        assert(d_executor_ptr);
        auto span = std::span<std::experimental::coroutine_handle<>>(d_storage.data(), d_storage.size());
        d_executor_ptr->schedule_all(span);
        d_storage.clear();
    }

    void park(Executor& e, std::experimental::coroutine_handle<> h)
    {
        COROBATCH_LOG_TRACE << "Parking the handle. Already parked: " << d_storage.size();
        if (d_executor_ptr == nullptr)
        {
            d_executor_ptr = &e;
        }
        assert(d_executor_ptr == &e &&
               "All the tasks must be executed on the same executor to use the SingleExecutorRescheduler");
        assert(d_num_pending < MaxConcurrentTasksWaitingOnBatch &&
               "The number of tasks waiting on the batch was more than the maximum supported");
        d_storage.push_back(h);
    }

    std::size_t num_pending() const { return d_storage.size(); }

    bool empty() const { return d_storage.empty(); }

private:
    Executor* d_executor_ptr = nullptr;
    Storage d_storage;
};

template<typename Executor>
SingleExecutorRescheduler(Executor& e)
    -> SingleExecutorRescheduler<Executor, std::vector<std::experimental::coroutine_handle<>>>;

template<std::size_t S, typename Executor>
SingleExecutorRescheduler<Executor, StaticVector<S>> fixedSingleExecutorRescheduler(Executor& e)
{
    return SingleExecutorRescheduler<Executor, StaticVector<S>>(e);
}

namespace private_ {

template<typename T, std::size_t S>
class CircularBuff
{
private:
    static constexpr std::size_t t_size = sizeof(T);
    static constexpr std::size_t max_size = t_size * S;

    alignas(T) std::array<char, max_size> d_mem;
    std::size_t d_begin = 0;
    std::size_t d_end = 0;
    std::size_t d_size = 0;

public:
    CircularBuff() = default;

    CircularBuff(const CircularBuff& o) { insert(end(), o.begin(), o.end()); }

    CircularBuff(CircularBuff&& o)
    {
        insert(end(), std::make_move_iterator(o.begin()), std::make_move_iterator(o.end()));
    }

    ~CircularBuff()
    {
        for (T& obj : *this)
        {
            obj.~T();
        }
        d_begin = 0;
        d_end = 0;
        d_size = 0;
    }

    bool empty() const { return d_size == 0; }
    std::size_t size() const { return d_size; }

    T* begin() { return reinterpret_cast<T*>(d_mem.data() + d_begin); }
    T* end() { return reinterpret_cast<T*>(d_mem.data() + d_end); }

    T& front()
    {
        assert(not empty());
        return *begin();
    }

    void pop_front()
    {
        begin()->~T();
        d_begin = (d_begin + t_size) % max_size;
        d_size--;
    }

    template<typename It>
    void insert([[maybe_unused]] T* pos, It b, It e)
    {
        assert(pos == end() && "The buffer only allows to insert at the end of it");
        assert(size() + std::distance(b, e) <= S);
        for (It c = b; c != e; ++c)
        {
            new (static_cast<void*>(end())) T(*c);
            d_end = (d_end + t_size) % max_size;
            d_size++;
        }
    }
};

} // namespace private_

template<std::size_t MaxConcurrentScheduledCoros>
class FixedSizeExecutor
{
public:
    FixedSizeExecutor() = default;
    FixedSizeExecutor(const FixedSizeExecutor&) = delete;
    FixedSizeExecutor& operator=(const FixedSizeExecutor&) = delete;

    void run()
    {
        while (not d_ready_coroutines.empty())
        {
            std::experimental::coroutine_handle<> next = d_ready_coroutines.front();
            d_ready_coroutines.pop_front();
            next.resume();
        }
    }

    ~FixedSizeExecutor() { assert(d_ready_coroutines.empty()); }

    void schedule_all(std::span<std::experimental::coroutine_handle<>> new_coros)
    {
        d_ready_coroutines.insert(d_ready_coroutines.end(), new_coros.begin(), new_coros.end());
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
    private_::CircularBuff<std::experimental::coroutine_handle<>, MaxConcurrentScheduledCoros> d_ready_coroutines;
};

} // namespace corobatch

#endif

//////////////////////////////////////////////////////////////////////
// End file: corobatch/utility/executor.hpp
//////////////////////////////////////////////////////////////////////
