#pragma once

#include <algorithm>
#include <cassert>
#include <experimental/coroutine>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include <corobatch/executor.hpp>
#include <corobatch/private_/log.hpp>

#define MY_FWD(...) ::std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)
namespace corobatch {

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

template<typename It> // It of IBatcher*
bool force_execution(It begin, It end) // requires requires (It it) { *it -> std::convertible_to<IBatcher*>}
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

template<typename It> // It of IBatcher*
bool has_any_pending(It begin, It end) // requires requires (It it) { *it -> std::convertible_to<IBatcher*>}
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

template<typename Batcher, typename AccumulationStorage, typename... Args>
auto batcher_record_arguments(Batcher& b, AccumulationStorage& as, ArgTypeList<Args...>)
    -> decltype(b.record_arguments(as, std::declval<Args&&>()...));

} // namespace private_

template<typename T, typename NoRefT = std::remove_reference_t<T>>
concept ConceptBatcher = requires(T batcher,
                                  typename NoRefT::AccumulationStorage accumulation_storage,
                                  typename NoRefT::ExecutedResults executed_result,
                                  typename NoRefT::Handle handle,
                                  typename NoRefT::Args args,
                                  typename NoRefT::ResultType result_type,
                                  std::function<void(typename NoRefT::ExecutedResults)> ondone_callback)
{
    {
        args
    }
    ->private_::ConceptArgTypeList;
    {
        batcher.get_accumulation_storage()
    }
    ->private_::ConceptIsSame<typename NoRefT::AccumulationStorage>;
    {
        batcher_record_arguments(batcher, accumulation_storage, args)
    }
    ->private_::ConceptIsSame<typename NoRefT::Handle>;
    {
        batcher.execute(std::move(accumulation_storage), std::move(ondone_callback))
    }
    ->private_::ConceptIsSame<typename NoRefT::ExecutedResults>;
    {
        batcher.get_result(std::move(handle), executed_result)
    }
    ->private_::ConceptIsSame<typename NoRefT::ResultType>;
    {
        batcher.must_execute(std::as_const(accumulation_storage))
    }
    ->private_::ConceptIsSame<bool>;
};

// Batcher implementation format

/*
struct Batcher {
    using AccumulationStorage = ...;
    using ExecutedResults = ...;
    using Handle = ...;
    using Args = ArgTypeList<...>;
    using ResultType = ...;

    AccumulationStorage get_accumulation_storage();
    Handle record_arguments(AccumulationStorage& , Args&&...);
    void execute(AccumulationStorage&&, std::function<void(ExecutedResults)>);
    ResultType get_result(Handle, ExecutedResults&);
    bool must_execute(const AccumulationStorage&);
};
*/

namespace private_ {

template<ConceptBatcher Batcher, typename ArgsList>
class BatcherWrapperImpl;

template<ConceptBatcher Batcher, typename... Args>
class BatcherWrapperImpl<Batcher, ArgTypeList<Args...>> : public IBatcher
{
private:
    using NoRefBatcher = std::remove_reference_t<Batcher>;

    struct Batch
    {
        Batch(Batcher& batcher) : d_batcher(batcher), d_storage(d_batcher.get_accumulation_storage())
        {
            COROBATCH_LOG_DEBUG << "New batch created";
        }
        ~Batch() { assert(d_waiting_coros.empty()); }

        Batcher& d_batcher;
        typename NoRefBatcher::AccumulationStorage d_storage;
        std::optional<typename NoRefBatcher::ExecutedResults> d_result;
        std::vector<std::experimental::coroutine_handle<>> d_waiting_coros;
    };

    struct Awaitable
    {
        bool await_ready() { return d_batch->d_result.has_value(); }

        decltype(auto) await_resume()
        {
            assert(await_ready());
            decltype(auto) result = d_batch->d_batcher.get_result(d_batcher_handle, d_batch->d_result.value());
            COROBATCH_LOG_DEBUG << "Resuming coro " << private_::PrintIfPossible(result);
            return MY_FWD(result);
        }

        auto await_suspend(std::experimental::coroutine_handle<> h) { d_batch->d_waiting_coros.push_back(h); }

        // private:
        typename NoRefBatcher::Handle d_batcher_handle;
        std::shared_ptr<Batch> d_batch;
    };

public:
    BatcherWrapperImpl(Executor& executor, Batcher batcher)
    : d_executor(executor), d_batcher(MY_FWD(batcher)), d_current_batch(make_new_batch())
    {
    }

    ~BatcherWrapperImpl()
    {
        assert(d_current_batch->d_waiting_coros.empty() &&
               "Force the execution of the batch if it has any pending coroutines");
    }

    Awaitable operator()(Args... args)
    {
        COROBATCH_LOG_DEBUG_BLOCK
        {
            COROBATCH_LOG_STREAM << "Recording parameter";
            ((COROBATCH_LOG_STREAM << ' ' << private_::PrintIfPossible(args)), ...);
        }
        typename NoRefBatcher::Handle batcherHandle =
            d_batcher.record_arguments(d_current_batch->d_storage, MY_FWD(args)...);
        Awaitable awaitable{batcherHandle, d_current_batch};
        if (d_batcher.must_execute(d_current_batch->d_storage))
        {
            executeBatch();
        }
        return awaitable;
    }

    int getNumPendingCoros() const override { return d_current_batch->d_waiting_coros.size(); }

    void executeBatch() override
    {
        COROBATCH_LOG_DEBUG << "Executing batch";
        d_batcher.execute(std::move(d_current_batch->d_storage),
                          [this, currbatch = d_current_batch](typename NoRefBatcher::ExecutedResults results) mutable {
                              assert(not currbatch->d_waiting_coros.empty() && "Did you call the callback twice?");
                              currbatch->d_result = std::move(results);
                              this->d_executor.schedule_all(currbatch->d_waiting_coros.begin(),
                                                            currbatch->d_waiting_coros.end());
                              currbatch->d_waiting_coros.clear();
                              COROBATCH_LOG_DEBUG << "Batch execution completed";
                          });
        d_current_batch = make_new_batch();
    }

private:
    Executor& d_executor;
    Batcher d_batcher;
    std::shared_ptr<Batch> d_current_batch;

    std::shared_ptr<Batch> make_new_batch() { return std::shared_ptr<Batch>(new Batch(d_batcher)); }
};

} // namespace private_

template<ConceptBatcher Batcher>
class BatcherWrapper : public private_::BatcherWrapperImpl<Batcher, typename std::remove_reference_t<Batcher>::Args>
{
private:
    using Base = private_::BatcherWrapperImpl<Batcher, typename std::remove_reference_t<Batcher>::Args>;

public:
    using Base::Base;
};

template<ConceptBatcher Batcher>
BatcherWrapper<Batcher> make_batcher(Executor& executor, Batcher&& batcher)
{
    return BatcherWrapper<Batcher>(executor, MY_FWD(batcher));
}

template<typename... Batchers>
auto make_batchers(Executor& executor, Batchers&&... batchers)
{
    return std::make_tuple(make_batcher(executor, MY_FWD(batchers))...);
}

} // namespace corobatch
#undef MY_FWD
