#pragma once

#include <cassert>
#include <corobatch/private_/log.hpp>
#include <deque>
#include <experimental/coroutine>

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

private:
    std::deque<std::experimental::coroutine_handle<>> d_ready_coroutines;
};

} // namespace corobatch
