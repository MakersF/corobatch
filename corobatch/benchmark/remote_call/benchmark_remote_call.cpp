#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <benchmark/benchmark.h>

#define COROBATCH_TRANSLATION_UNIT
#include <corobatch/corobatch.hpp>

using namespace std::chrono_literals;

constexpr std::size_t batch_size = 128;
// Fake object to simulate a class with a real size
using Object = std::array<char, 512>;

// Values from https://gist.github.com/jboner/2841832
constexpr auto datacenter_roundtrip = 500us;
constexpr auto trasmission_time_per_byte = 10ns;

void simulate_call_to_other_service(const std::string& data)
{
    std::this_thread::sleep_for(datacenter_roundtrip + trasmission_time_per_byte * data.size());
}

void serialize(std::string& s, const Object& o) { s.insert(s.end(), sizeof(Object), 'Z'); }

class RemoteCallBatcher
{
public:
    struct Storage
    {
        std::string data;
        std::size_t num_items = 0;
    };
    using AccumulationStorage = Storage;
    using ExecutedResults = bool;
    using Handle = std::nullptr_t;
    using Args = corobatch::ArgTypeList<Object>;
    using ResultType = bool;

    AccumulationStorage get_accumulation_storage() const { return {}; }

    Handle record_arguments(AccumulationStorage& s, const Object& val) const
    {
        serialize(s.data, val);
        s.num_items++;
        return nullptr;
    }

    template<typename Callback>
    void execute(AccumulationStorage&& s, Callback cb) const
    {
        simulate_call_to_other_service(s.data);
        cb(true);
    }

    ResultType get_result(Handle, ExecutedResults&) const { return true; }

    bool must_execute(const AccumulationStorage& s) const { return s.num_items >= batch_size; }
};

void send_data_batch(int num_requests)
{
    corobatch::Executor executor;
    RemoteCallBatcher remoteCallBatcher;
    auto sendRequest = corobatch::Batcher(remoteCallBatcher);

    auto action = [](auto& sendRequest) -> corobatch::task<void> {
        Object request;
        co_await sendRequest(request);
    };

    auto ignore = []() {};
    for (int i = 0; i < num_requests; i++)
    {
        corobatch::submit(executor, ignore, action(sendRequest));
    }

    corobatch::force_execution(sendRequest);
    executor.run();
}

void BM_send_data_no_batch(benchmark::State& state)
{
    for (auto _ : state)
    {
        for (int i = 0; i < state.range(0); i++)
        {
            Object request;
            std::string data;
            serialize(data, request);
            simulate_call_to_other_service(data);
        }
    }
}

void BM_send_data_batch(benchmark::State& state)
{
    corobatch::registerLoggerCb(corobatch::disabled_logger);
    for (auto _ : state)
    {
        send_data_batch(state.range(0));
    }
}

void BM_send_data_hand_batch(benchmark::State& state)
{
    for (auto _ : state)
    {
        std::string data;
        for (int i = 0; i < state.range(0); i++)
        {
            Object request;
            serialize(data, request);
            if (i % batch_size == batch_size - 1)
            {
                simulate_call_to_other_service(data);
                data.clear();
            }
        }
    }
}

BENCHMARK(BM_send_data_no_batch)->Range(8, 8 << 8);
BENCHMARK(BM_send_data_batch)->Range(8, 8 << 8);
BENCHMARK(BM_send_data_hand_batch)->Range(8, 8 << 8);
BENCHMARK_MAIN();
