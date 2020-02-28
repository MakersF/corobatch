#include "common.hpp"

std::vector<Bar3> foo1(const std::vector<Bar2>& in)
{
    std::vector<Bar3> res(in.size());
    return res;
}

std::vector<Bar5> foo2(const std::vector<Bar4>& in)
{
    std::vector<Bar5> res(in.size());
    return res;
}
