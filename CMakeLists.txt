# cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++-11 ..
cmake_minimum_required(VERSION 3.5)
project ( COROBATCH CXX )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options( -std=gnu++2a -fcoroutines-ts -nostdinc++ -isystem/usr/lib/llvm-11/include/c++/v1/)
SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -L/usr/lib/llvm-11/lib/ -lc++ -Wl,-rpath,/usr/lib/llvm-11/lib/")

add_library( common comparison/common.cpp )
target_include_directories( common PUBLIC comparision)

add_executable( nobatching comparison/nobatching.cpp )
target_include_directories( nobatching PUBLIC comparision)
target_link_libraries( nobatching PRIVATE common)

add_executable( handbatching comparison/handbatching.cpp )
target_include_directories( handbatching PUBLIC comparision)
target_link_libraries( handbatching PRIVATE common)

add_executable( corobatching comparison/corobatching.cpp )
target_include_directories( corobatching PUBLIC comparision include)
target_link_libraries( corobatching PRIVATE common )

add_executable( main main.m.cpp )
target_include_directories( main PRIVATE include )
