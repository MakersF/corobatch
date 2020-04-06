#include "common.hpp"

#define COROBATCH_TRANSLATION_UNIT
#include <corobatch/corobatch.hpp>

std::vector<std::string> corobatching(const std::vector<Bar1>& bar1s)
{
    auto logic = [](const Bar1& bar1, auto&& foo1, auto&& foo2) -> corobatch::task<std::string> {
        Bar2 bar2 = bar1.baz1;
        Bar3 bar3 = co_await foo1(bar2);
        Bar4 bar4 = bar3.baz3;
        Bar5 bar5 = co_await foo2(bar4);
        Bar6 bar6 = bar5.baz5;
        co_return bar6.baz6;
    };

    auto foo1Accumulator = corobatch::syncVectorAccumulator<Bar3, Bar2>(foo1);
    auto foo2Accumulator = corobatch::syncVectorAccumulator<Bar5, Bar4>(foo2);
    auto [foo1Batcher, foo2Batcher] = corobatch::make_batchers(foo1Accumulator, foo2Accumulator);

    corobatch::Executor executor;
    std::size_t completed = 0;
    std::vector<std::string> res;
    auto onDone = [&res, &completed](std::string result) {
        res.push_back(std::move(result));
        completed++;
    };
    // Note: the result is not guaranteed in order
    for (const Bar1& bar : bar1s)
    {
        corobatch::submit(executor, onDone, logic(bar, foo1Batcher, foo2Batcher));
    }

    while (completed != bar1s.size())
    {
        executor.run();
        corobatch::force_execution(foo1Batcher, foo2Batcher);
    }
    return res;
}

int main() { return 0; }
