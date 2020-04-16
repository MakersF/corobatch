#ifndef COROBATCH_UTILITY_EXECUTOR_HPP
#define COROBATCH_UTILITY_EXECUTOR_HPP

#include <array>
#include <cassert>
#include <corobatch/private_/log.hpp>

namespace corobatch {

// Can be used when all the tasks are scheduled on the same executor
template<typename Executor, std::size_t MaxConcurrentTasksWaitingOnBatch>
class SingleExecutorRescheduler
{
public:
    SingleExecutorRescheduler() = default;
    SingleExecutorRescheduler(Executor& e) : d_executor_ptr(&e) {}

    void reschedule()
    {
        assert(d_executor_ptr);
        d_executor_ptr->schedule_all({d_waiting_coros.begin(), d_waiting_coros.begin() + d_num_pending});
        d_num_pending = 0;
    }

    void park(Executor& e, std::experimental::coroutine_handle<> h)
    {
        COROBATCH_LOG_TRACE << "Parking the handle. Already parked: " << d_num_pending;
        if (d_executor_ptr == nullptr)
        {
            d_executor_ptr = &e;
        }
        assert(d_executor_ptr == &e &&
               "All the tasks must be executed on the same executor to use the SingleExecutorRescheduler");
        assert(d_num_pending < MaxConcurrentTasksWaitingOnBatch &&
               "The number of tasks waiting on the batch was more than the maximum supported");
        d_waiting_coros[d_num_pending] = h;
        d_num_pending++;
    }

    std::size_t num_pending() const { return d_num_pending; }

    bool empty() const { return d_num_pending == 0; }

private:
    Executor* d_executor_ptr = nullptr;
    std::array<std::experimental::coroutine_handle<>, MaxConcurrentTasksWaitingOnBatch> d_waiting_coros;
    std::size_t d_num_pending = 0;
};

template<std::size_t S, typename Executor>
SingleExecutorRescheduler<Executor, S> singleExecutorRescheduler(Executor& e)
{
    return SingleExecutorRescheduler<Executor, S>(e);
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
