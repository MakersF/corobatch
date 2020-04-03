#pragma once

#include <functional>
#include <memory>
#include <sstream>

#include <boost/asio.hpp>

#include <nop/base/encoding.h>
#include <nop/serializer.h>
#include <nop/utility/buffer_reader.h>
#include <nop/utility/die.h>
#include <nop/utility/stream_writer.h>

template<typename T>
void send_message(boost::asio::ip::tcp::socket& socket,
                  T&& obj,
                  std::function<void()> onSuccess,
                  std::function<void()> onError)
{
    using Writer = nop::StreamWriter<std::ostringstream>;
    nop::Serializer<Writer> serializer;
    auto status = serializer.Write(obj);
    if (!status)
    {
        std::cout << "Error serializing = " << status.GetErrorMessage() << std::endl;
        onError();
        return;
    }
    auto buffer = std::make_unique<std::string>(serializer.take().take().str());
    auto buffer_size = std::make_unique<std::size_t>(buffer->size());

    auto data_to_write = boost::asio::buffer(static_cast<void*>(buffer_size.get()), sizeof(std::size_t));
    boost::asio::async_write(
        socket,
        data_to_write,
        [&socket, buffer = std::move(buffer), buffer_size = std::move(buffer_size), onSuccess, onError](
            boost::system::error_code ec, std::size_t /*length*/) mutable {
            if (!ec)
            {
#ifdef COROBATCH_EXAMPLES_LOGGING_EXTRA
                std::cout << "Payload size written = " << *buffer_size << std::endl;
#endif
                auto data_to_write = boost::asio::buffer(*buffer);
                boost::asio::async_write(socket,
                                         data_to_write,
                                         [buffer = std::move(buffer), onSuccess, onError](boost::system::error_code ec,
                                                                                          std::size_t /*length*/) {
                                             if (!ec)
                                             {
#ifdef COROBATCH_EXAMPLES_LOGGING_EXTRA
                                                 std::cout << "Payload written =";
                                                 for (char c : *buffer)
                                                 {
                                                     printf(" %02hhx", c);
                                                 }
                                                 std::cout << std::endl;
#endif
                                                 onSuccess();
                                             }
                                             else
                                             {
                                                 std::cout << "Error " << __LINE__ << " " << ec << std::endl;
                                                 onError();
                                             }
                                         });
            }
            else
            {
                std::cout << "Error " << __LINE__ << " " << ec << std::endl;
                onError();
            }
        });
}

template<typename T>
void receive_message(boost::asio::ip::tcp::socket& socket,
                     std::function<void(T)> onSuccess,
                     std::function<void()> onError)
{
    auto size = std::make_unique<std::size_t>(0);
    auto data_to_fill = boost::asio::buffer(static_cast<void*>(size.get()), sizeof(std::size_t));
    boost::asio::async_read(
        socket,
        data_to_fill,
        [&socket, size = std::move(size), onSuccess, onError](boost::system::error_code ec, std::size_t /*length*/) {
            if (!ec)
            {
#ifdef COROBATCH_EXAMPLES_LOGGING_EXTRA
                std::cout << "Payload size read = " << *size << std::endl;
#endif
                auto buffer = std::make_unique<char[]>(*size);
                auto data_to_fill = boost::asio::buffer(buffer.get(), *size);
                boost::asio::async_read(
                    socket,
                    data_to_fill,
                    [buffer = std::move(buffer), buffer_size = *size, onSuccess, onError](boost::system::error_code ec,
                                                                                          std::size_t /*length*/) {
                        if (!ec)
                        {
#ifdef COROBATCH_EXAMPLES_LOGGING_EXTRA
                            std::cout << "Payload read =";
                            for (const char *begin = buffer.get(), *end = buffer.get() + buffer_size; begin != end;
                                 begin++)
                            {
                                printf(" %02hhx", *begin);
                            }
                            // for (char c : *buffer)
                            // {
                            //     // std::cout << " " << static_cast<int>(c);
                            //     printf(" %02hhx", c);
                            // }
                            std::cout << std::endl;
#endif
                            nop::Deserializer<nop::BufferReader> deserializer(buffer.get(), buffer_size);
                            T obj;
                            auto status = deserializer.Read(&obj);
                            if (!status)
                            {
                                std::cout << "Error deserializing = " << status.GetErrorMessage() << std::endl;
                                onError();
                                return;
                            }
                            onSuccess(std::move(obj));
                        }
                        else
                        {
                            std::cout << "Error " << __LINE__ << " " << ec << std::endl;
                            onError();
                        }
                    });
            }
            else
            {
                std::cout << "Error " << __LINE__ << " " << ec << std::endl;
                onError();
            }
        });
}
