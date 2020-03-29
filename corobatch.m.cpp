#include <algorithm>
#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <corobatch/corobatch.hpp>

std::vector<std::thread> threads;
constexpr auto executor = [](auto&& f, auto&&... args) {
    threads.emplace_back(std::forward<decltype(f)>(f), std::forward<decltype(args)>(args)...);
};

int main()
{
    std::vector<int> data = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};

    auto action = [](int v, auto&& int2dbl, auto&& dblint2str) -> corobatch::task<std::string> {
        double r = co_await int2dbl(v);
        try {
            auto s = co_await dblint2str(r, v);
            co_return s + co_await dblint2str(r, v);
        } catch (std::exception& e) {
            co_return e.what();
        }
    };

    auto int2dbl =
        corobatch::SizedBatcher(corobatch::syncVectorBatcher<double, int>([](const std::vector<int>& params) {
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

    auto dblint2str = corobatch::WaitableBatcher(
        corobatch::vectorBatcher<std::string, double, int>(
            executor,
            [](const std::vector<std::tuple<double, int>>& params, std::function<void(std::vector<std::string>)> cb) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3000));
                std::vector<std::string> res;
                for (auto&& [dbl, integer] : params)
                {
                    std::string val = std::to_string(dbl) + "_" + std::to_string(integer);
                    res.push_back(val);
                }
                throw std::runtime_error("Error!!");
                cb(res);
            }),
        waitState);

    corobatch::Executor executor;
    auto [int2dbl_conv, dblint2str_conv] = corobatch::make_batcher(executor, int2dbl, dblint2str);

    int uncompleted = 0;
    for (auto it = data.begin(); it != data.end(); ++it)
    {
        uncompleted++;
        corobatch::submit(
            [it, &uncompleted](std::string result) {
                std::cout << *it << "=" << result << " ";
                uncompleted--;
            },
            action(*it, int2dbl_conv, dblint2str_conv));
    }

    while (uncompleted != 0)
    {
        waitState.wait_for_completion();
        executor.run();
        corobatch::force_execution(int2dbl_conv, dblint2str_conv);
    }

    std::cout << "Joining" << std::endl;
    for (auto& t : threads)
    {
        t.join();
    }
    std::cout << std::endl;
    return 0;
}