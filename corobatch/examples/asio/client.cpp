#include <iostream>
#include <vector>

#include <boost/asio.hpp>

#define COROBATCH_TRANSLATION_UNIT
#include <corobatch/corobatch.hpp>

#include "communicate.hpp"

using namespace boost;
using asio::ip::tcp;

// Send the request over TCP to the specified endpoints and call the callback when the result is received
void rpc_power_of_2(asio::io_context& io_context,
                    const tcp::resolver::results_type& endpoints,
                    std::vector<int> request,
                    std::function<void(std::vector<int>)> cb)
{
    // Connect to the end points, send the request, wait for the response to be deliveded, and invoke the callback once
    // done. We need to capture the socket_ptr in the lambdas to make sure the socket remains alive. We can't keep it on
    // the stack because all the functions return immediately.
    auto socket_ptr = std::make_shared<tcp::socket>(io_context);
    asio::async_connect(*socket_ptr, endpoints, [socket_ptr, request, cb](system::error_code ec, tcp::endpoint) {
        if (!ec)
        {
            send_message(
                *socket_ptr,
                request,
                [socket_ptr, cb]() {
                    std::cout << "Request sent" << std::endl;
                    receive_message<std::vector<int>>(
                        *socket_ptr,
                        [socket_ptr, cb](std::vector<int> response) {
                            std::cout << "Response received" << std::endl;
                            socket_ptr->close();
                            cb(response);
                        },
                        [socket_ptr]() {
                            std::cout << "Error receiving message " << std::endl;
                            socket_ptr->close();
                        });
                },
                [socket_ptr]() {
                    std::cout << "Error sending message " << std::endl;
                    socket_ptr->close();
                });
        }
        else
        {
            std::cout << "Error connecting " << ec << std::endl;
            socket_ptr->close();
        }
    });
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: client <host> <port>\n";
        return 1;
    }

    corobatch::registerLoggerCb(corobatch::debug_logger);

    asio::io_context io_context;
    tcp::resolver resolver{io_context};
    auto endpoints = resolver.resolve(argv[1], argv[2]);

    corobatch::Executor executor;
    std::vector<corobatch::IBatcher*> batchers;

    // When the accumulator executes, it will immediately invoke rpc_power_of_2. Once we receive the data back,
    // asio calls the function to notify of the received data and we schedule the executor to run again,
    // since the received response will have unblocked some tasks.
    auto rpc_power_of_2_accumulator = corobatch::vectorAccumulator<int, int>(
        corobatch::immediate_invoke, [&](std::vector<int> params, std::function<void(std::vector<int>)> cb) {
            // cb should be called with the result of processing 'params'
            rpc_power_of_2(io_context, endpoints, params, [&, cb](std::vector<int> response) {
                cb(response);
                asio::post(io_context, [&]() {
                    // We have been notified of the result, so some tasks have been unblocked.
                    // We can schedule to continue the execution.
                    executor.run();
                    // The execution might have resulted in some tasks being blocked again,
                    // so we force the execution to unblock them.
                    corobatch::force_execution(batchers.begin(), batchers.end());
                });
            });
        });

    // Prepare the batcher to be used with the given executor inside the coroutine
    auto rpc_power_of_2_batcher = corobatch::Batcher(rpc_power_of_2_accumulator);
    batchers.push_back(std::addressof(rpc_power_of_2_batcher));

    // Function to create the task
    auto make_task = [&rpc_power_of_2_batcher](int elem) -> corobatch::task<int> {
        std::cout << "Starting: " << elem << std::endl;
        int pow2 = co_await rpc_power_of_2_batcher(elem);
        std::cout << "Resuming 1: " << elem << std::endl;
        int pow4 = co_await rpc_power_of_2_batcher(pow2);
        std::cout << "Resuming 2: " << elem << std::endl;
        co_return pow4 - pow2;
    };

    // Execute the task for each of the data
    std::vector<int> data_to_process = {1, 2, 3, 4};
    for (int i : data_to_process)
    {
        corobatch::submit(
            executor, [i](int result) { std::cout << i << " = " << result << std::endl; }, make_task(i));
    }

    // The tasks eagerly started executing, and they might be blocked.
    // We force the execution of the batch to unblock them
    corobatch::force_execution(batchers.begin(), batchers.end());

    // Start the main loop that will drive the execution of the whole program
    io_context.run();
}
