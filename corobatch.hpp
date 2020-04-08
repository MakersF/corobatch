// Generated on Wed  8 Apr 10:22:45 BST 2020
// Commit: f58ffef5f5a79d9608fa90d987cfd035e617a7d0

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
    DEBUG = 64,
    INFO = 128,
    ERROR = 256
};

// Register logger
using LoggerCb = std::function<std::ostream*(LogLevel)>;

extern LoggerCb disabled_logger;
extern LoggerCb debug_logger;
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

LoggerCb debug_logger = [](LogLevel level) -> std::ostream* {
    if (level >= LogLevel::DEBUG)
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

#define COROBATCH_LOG_DEBUG_BLOCK COROBATCH_LOG_BLOCK(::corobatch::LogLevel::DEBUG, "DEBUG")

#define COROBATCH_LOG_INFO_BLOCK COROBATCH_LOG_BLOCK(::corobatch::LogLevel::INFO, "INFO")

#define COROBATCH_LOG_ERROR_BLOCK COROBATCH_LOG_BLOCK(::corobatch::LogLevel::ERROR, "ERROR")

// Line macros

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
            const char* begin = reinterpret_cast<const char*>(&obj.d_value);
            const char* end = begin + sizeof(obj.d_value);
            std::ios_base::fmtflags previous_flags = os.flags();
            os << "[ ";
            for (; begin != end; begin++)
            {
                os << std::showbase << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*begin);
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

    template<typename It>
    void schedule_all(It begin, It end)
    {
        d_ready_coroutines.insert(d_ready_coroutines.end(), begin, end);
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
            executor_ptr->schedule_all(coroutines.begin(), coroutines.end());
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
            return MY_FWD(result);
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
#include <functional>
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

//////////////////////////////////////////////////////////////////////
// End file: corobatch/accumulate.hpp
//////////////////////////////////////////////////////////////////////

