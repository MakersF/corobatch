# cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++-11 -DCMAKE_CXX_FLAGS="-DBOOST_ASIO_DISABLE_CONCEPTS" ..
# The second define is required because of https://github.com/boostorg/asio/issues/312.
# To get the dependencies:
# 1. git clone https://github.com/google/libnop.git vendor/libnop
# 2. download boost https://sourceforge.net/projects/boost/files/boost/1.72.0/ and unpack it in vendor/boost.
#    There is no need to build boost. If you want change the code in the header to fix the issue above and
#    build with concepts enabled
# For benchmark, google benchmark is neeeded
# 1. git clone https://github.com/google/benchmark vendor/benchmark
# 2. cd vendor/benchmark && mkdir build && cd build
# 3. cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DCMAKE_CXX_COMPILER=/usr/bin/clang++-11 -DCMAKE_CXX_FLAGS="-pedantic -Wall -Wextra -std=gnu++2a -fcoroutines-ts -O3 -pthread -stdlib=libc++ -nostdinc++ -isystem/usr/lib/llvm-11/include/c++/v1/" -DCMAKE_CXX_LINK_FLAGS="-L/usr/lib/llvm-11/lib/ -lpthread -lc++ -Wl,-rpath,/usr/lib/llvm-11/lib/" ..
# 4. make
cmake_minimum_required(VERSION 3.5)
project ( COROBATCH CXX )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-pedantic -Wall -Wextra -std=gnu++2a -fcoroutines-ts -pthread -nostdinc++ -isystem/usr/lib/llvm-11/include/c++/v1/)
SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -L/usr/lib/llvm-11/lib/")

# add_compile_options( -fsanitize=address )
# SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -lasan")

SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -lpthread -lc++ -Wl,-rpath,/usr/lib/llvm-11/lib/")

add_compile_options( -O3 -g )
# add_compile_options(-g -O0 )


# add_compile_options( -flto )
# SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -flto -fuse-ld=lld-11")

add_library( corobatch INTERFACE )
target_include_directories( corobatch INTERFACE corobatch/include )

add_subdirectory(comparison)
add_subdirectory(corobatch/examples)
add_subdirectory(corobatch/benchmark)

add_executable( main corobatch.m.cpp )
target_link_libraries( main PRIVATE corobatch )
