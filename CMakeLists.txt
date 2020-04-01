# cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++-11 -DCMAKE_CXX_FLAGS="-DBOOST_ASIO_DISABLE_CONCEPTS" ..
# The second define is required because of https://github.com/boostorg/asio/issues/312.
# To get the dependencies:
# 1. git clone https://github.com/google/libnop.git vendor/libnop
# 2. download boost https://sourceforge.net/projects/boost/files/boost/1.72.0/ and unpack it in vendor/boost.
#    There is no need to build boost. If you want change the code in the header to fix the issue above and
#    build with concepts enabled
cmake_minimum_required(VERSION 3.5)
project ( COROBATCH CXX )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options( -pedantic -Wall -Wextra -std=gnu++2a -fcoroutines-ts -g -O0 -pthread -fsanitize=address -nostdinc++ -isystem/usr/lib/llvm-11/include/c++/v1/)
SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -L/usr/lib/llvm-11/lib/ -lasan -lpthread -lc++ -Wl,-rpath,/usr/lib/llvm-11/lib/")

add_library( corobatch INTERFACE )
target_include_directories( corobatch INTERFACE corobatch/include )

add_subdirectory(comparison)
add_subdirectory(corobatch/examples)

add_executable( main corobatch.m.cpp )
target_link_libraries( main PRIVATE corobatch )