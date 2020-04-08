#include <algorithm>
#include <atomic>
#include <iostream>
#include <string>
#include <vector>

#define COROBATCH_TRANSLATION_UNIT
//#include <corobatch/corobatch.hpp>
#include "corobatch.hpp"

int main()
{
    corobatch::registerLoggerCb(corobatch::debug_logger);

    std::vector<int> data = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

    auto action = [](int v, auto&& int2double, auto&& double_int2string) -> corobatch::task<std::string> {
        double r = co_await int2double(v);
        try {
            auto s = co_await double_int2string(r, v);
            co_return s + co_await double_int2string(r, v);
        } catch (std::exception& e) {
            co_return e.what();
        }
    };

    auto int2double =
        corobatch::SizedAccumulator(corobatch::syncVectorAccumulator<double, int>([](const std::vector<int>& params) {
                                    std::vector<double> res;
                                    for (int v : params)
                                    {
                                        double val = v + 0.5;
                                        res.push_back(val);
                                    }
                                    return res;
                                }),
                                3);

    corobatch::MTWaitState waitState;

    corobatch::InvokeOnThread invokeOnThread;
    auto double_int2string = corobatch::WaitableAccumulator(
        corobatch::vectorAccumulator<std::string, double, int>(
            std::ref(invokeOnThread),
            [](const std::vector<std::tuple<double, int>>& params, std::function<void(std::vector<std::string>)> cb) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                std::vector<std::string> res;
                for (auto&& [dbl, integer] : params)
                {
                    std::string val = std::to_string(dbl) + "_" + std::to_string(integer);
                    res.push_back(val);
                }
                //throw std::runtime_error("Error!!");
                cb(res);
            }),
        waitState);

    corobatch::Executor executor;
    auto [int2double_conv, double_int2string_conv] = corobatch::make_batchers(int2double, double_int2string);

    int uncompleted = 0;
    for (auto it = data.begin(); it != data.end(); ++it)
    {
        uncompleted++;
        corobatch::submit(executor,
            [it, &uncompleted](std::string result) {
                std::cout << *it << "=" << result << " ";
                uncompleted--;
            },
            action(*it, int2double_conv, double_int2string_conv));
    }

    while (uncompleted != 0)
    {
        waitState.wait_for_completion();
        executor.run();
        corobatch::force_execution(int2double_conv, double_int2string_conv);
    }

    std::cout << std::endl;
    return 0;
}
