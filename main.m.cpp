#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <corobatch.hpp>

int main()
{
    std::vector<int> data = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

    auto action = [](int v, auto&& int2dbl, auto&& dblint2str) -> corobatch::task<std::string> {
        double r = co_await int2dbl(v);
        auto s = co_await dblint2str(r, v);
        co_return s + co_await dblint2str(r, v);
    };

    auto int2dbl = corobatch::vectorBatcher<double, int>([](const std::vector<int>& params) {
        std::vector<double> res;
        for (int v : params)
        {
            double val = v + 0.5;
            res.push_back(val);
        }
        return res;
    });

    auto dblint2str =
        corobatch::vectorBatcher<std::string, double, int>([](const std::vector<std::tuple<double, int>>& params) {
            std::vector<std::string> res;
            for (auto&& [dbl, integer] : params)
            {
                std::string val = std::to_string(dbl) + "_" + std::to_string(integer);
                res.push_back(val);
            }
            return res;
        });

    auto onComputed = [](std::vector<int>::iterator it, std::string result) {
        std::cout << *it << "=" << result << " ";
    };

    corobatch::batch(data.begin(), data.end(), onComputed, action, int2dbl, dblint2str);

    std::cout << std::endl;
    return 0;
}
