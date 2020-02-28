#include "common.hpp"
#include <algorithm>

std::vector<std::string> hand_batching(const std::vector<Bar1>& bar1s)
{
    std::vector<Bar2> bar2s;
    std::transform(bar1s.begin(), bar1s.end(), std::back_inserter(bar2s), [](const Bar1& bar1) { return bar1.baz1; });
    std::vector<Bar3> bar3s = foo1(bar2s);
    std::vector<Bar4> bar4s;
    std::transform(bar3s.begin(), bar3s.end(), std::back_inserter(bar4s), [](const Bar3& bar3) { return bar3.baz3; });
    std::vector<Bar5> bar5s = foo2(bar4s);
    std::vector<Bar6> bar6s;
    std::transform(bar5s.begin(), bar5s.end(), std::back_inserter(bar6s), [](const Bar5& bar5) { return bar5.baz5; });
    std::vector<std::string> bar7s;
    std::transform(bar6s.begin(), bar6s.end(), std::back_inserter(bar7s), [](const Bar6& bar6) { return bar6.baz6; });
    return bar7s;
}

int main() {
    return 0;
}
