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

#if 0
template<typename T>
concept Batcher = requires {
    typename T::AccumulationStorage;
    typename T::ExecutedResults;
    typename T::Handle;
    typename T::Args; // TODO must be an instance of ArgTypeList
    typename T::ResultType;

    // TODO check the expected methods
};

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
    BatcherWrapper(Executor& executor, Batcher batcher)
    : d_executor(executor), d_batcher(MY_FWD(batcher)), d_current_batch(make_new_batch())
    {
    }

    ~BatcherWrapper()
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

namespace private_ {

template<typename Batcher, typename... Args>
BatcherWrapper<Batcher, Args...>
    construct_batcherwrapper_impl(Executor& executor, Batcher&& batcher, ArgTypeList<Args...>)
{
    return BatcherWrapper<Batcher, Args...>(executor, MY_FWD(batcher));
}

} // namespace private_

template<typename Batcher>
auto make_batcher(Executor& executor, Batcher&& batcher)
{
    using Args = typename std::remove_reference_t<Batcher>::Args;
    return private_::construct_batcherwrapper_impl<Batcher>(executor, MY_FWD(batcher), Args{});
}

template<typename... Batchers>
auto make_batchers(Executor& executor, Batchers&&... batchers)
{
    return std::make_tuple(make_batcher(executor, MY_FWD(batchers))...);
}

} // namespace corobatch
#undef MY_FWD
