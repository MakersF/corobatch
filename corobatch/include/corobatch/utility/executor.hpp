#ifndef COROBATCH_UTILITY_EXECUTOR_HPP
#define COROBATCH_UTILITY_EXECUTOR_HPP

#include <array>
#include <cassert>
#include <corobatch/private_/log.hpp>

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
