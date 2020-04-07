#include <iostream>
#include <vector>

#include <boost/asio.hpp>

#include "communicate.hpp"

using boost::asio::ip::tcp;

void async_accept(tcp::acceptor& acceptor)
{
    acceptor.async_accept([&acceptor](boost::system::error_code ec, tcp::socket socket) {
        if (!ec)
        {
            std::cout << "Accepted connection" << std::endl;
            auto socket_ptr = std::make_shared<tcp::socket>(std::move(socket));
            receive_message<std::vector<int>>(
                *socket_ptr,
                [socket_ptr](std::vector<int> request) {
                    std::cout << "Request received. size = " << request.size() << " value =";
                    std::vector<int> response;
                    for (const int& n : request)
                    {
                        std::cout << " " << n;
                        response.push_back(n * n);
                    }
                    std::cout << std::endl;
                    send_message(
                        *socket_ptr, response, [socket_ptr]() { std::cout << "Response sent" << std::endl; }, []() {});
                },
                []() {});
        }
        async_accept(acceptor);
    });
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: server <port>\n";
        return 1;
    }

    boost::asio::io_context io_context;
    tcp::endpoint endpoint(tcp::v4(), std::atoi(argv[1]));
    tcp::acceptor acceptor(io_context, endpoint);
    async_accept(acceptor);
    io_context.run();
}
