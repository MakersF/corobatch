#include <vector>
#include <iostream>

#include <boost/asio.hpp>
#include <corobatch/corobatch.hpp>

#include "communicate.hpp"

using namespace boost;
using asio::ip::tcp;

// Send the request over TCP to the specified endpoints and call the callback when the result is received
void rpc_power_of_2(asio::io_context& io_context, const tcp::resolver::results_type& endpoints, std::vector<int> request, std::function<void(std::vector<int>)> cb)
{
    // Connect to the end points, send the request, wait for the response to be deliveded, and invoke the callback once done.
    // We need to capture the socket_ptr in the lambdas to make sure the socket remains alive.
    // We can't keep it on the stack because all the functions return immediately.
    auto socket_ptr = std::make_shared<tcp::socket>(io_context);
    asio::async_connect(*socket_ptr, endpoints,
        [socket_ptr, request, cb](system::error_code ec, tcp::endpoint)
        {
            if (!ec)
            {
                send_message(*socket_ptr, request, [socket_ptr, cb]() {
                    std::cout << "Request sent" << std::endl;
                    receive_message<std::vector<int>>(*socket_ptr, [socket_ptr, cb](std::vector<int> response) {
                        std::cout << "Response received" << std::endl;
                        socket_ptr->close();
                        cb(response);
                    }, [socket_ptr](){
                        std::cout << "Error receiving message " << std::endl;
                        socket_ptr->close();
                    });
                }, [socket_ptr](){
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

    std::vector<int> request = {1,2,3,4};

    asio::io_context io_context;
    tcp::resolver resolver{io_context};
    auto endpoints = resolver.resolve(argv[1], argv[2]);

    corobatch::Executor executor;

    // When the batcher executes, it will immediately invoke rpc_power_of_2. Once we receive the data back,
    // we notify the batcher of the received data and we schedule the executor to run again,
    // since the received will have unblocked some tasks.
    auto rpc_power_of_2_batcher = corobatch::vectorBatcher<int, int>(
            corobatch::immediate_invoke,
            [&](std::vector<int> params, std::function<void(std::vector<int>)> cb) {
                // cb should be called with the result of processisng 'params'
                rpc_power_of_2(io_context, endpoints, params, [&, cb](std::vector<int> response) {
                    cb(response);
                    // We have notified of the result, so some tasks have been unblocked. We can schedule to continue the execution
                    asio::post(io_context, [&executor](){
                        executor.run();
                    });
                });
            });

    auto rpc_power_of_2_batch_conv = std::get<0>(corobatch::make_batcher(executor, rpc_power_of_2_batcher));

    auto make_task = [&rpc_power_of_2_batch_conv](int elem) -> corobatch::task<int> {
        int pow2 = co_await rpc_power_of_2_batch_conv(elem);
        co_return pow2;
    };

    std::vector<int> data_to_process = {1,2,3,4};
    for(int i : data_to_process) {
        corobatch::submit(
            [i](int result) {
                std::cout << i << " * " << i << " = " << result << std::endl;
            }, make_task(i));
    }

    while(corobatch::has_any_pending(rpc_power_of_2_batch_conv)) {
        corobatch::force_execution(rpc_power_of_2_batch_conv);
    }

    io_context.run();
}
