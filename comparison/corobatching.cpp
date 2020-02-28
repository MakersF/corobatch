#include "common.hpp"

#include <corobatch.hpp>

std::vector<std::string> no_batching(const std::vector<Bar1>& bar1s)
{
    auto logic = [](const Bar1& bar1, auto&& foo1, auto&& foo2) -> corobatch::task<std::string> {
        Bar2 bar2 = bar1.baz1;
        Bar3 bar3 = co_await foo1(bar2);
        Bar4 bar4 = bar3.baz3;
        Bar5 bar5 = co_await foo2(bar4);
        Bar6 bar6 = bar5.baz5;
        co_return bar6.baz6;
    };
    std::vector<std::string> res;
    // Note: the result is not in order
    corobatch::batch(bar1s.begin(), bar1s.end(),
                     [&res](std::vector<Bar1>::const_iterator, std::string&& s) { res.push_back(std::move(s));},
                     logic,
                     corobatch::vectorBatcher<Bar3, Bar2>(foo1),
                     corobatch::vectorBatcher<Bar5, Bar4>(foo2));
    return res;
}

int main() {
    return 0;
}
