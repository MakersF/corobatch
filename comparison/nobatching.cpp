#include "common.hpp"

std::vector<std::string> no_batching(const std::vector<Bar1>& bar1s)
{
    std::vector<std::string> res;
    for(const Bar1& bar1 : bar1s) {
        Bar2 bar2 = bar1.baz1;
        Bar3 bar3 = foo1({bar2}).at(0);
        Bar4 bar4 = bar3.baz3;
        Bar5 bar5 = foo2({bar4}).at(0);
        Bar6 bar6 = bar5.baz5;
        res.push_back(bar6.baz6);
    }
    return res;
}

int main() {
    return 0;
}
