add_library( boost INTERFACE )
target_include_directories( boost INTERFACE  ${CMAKE_SOURCE_DIR}/vendor/boost )

add_library( nop INTERFACE )
target_include_directories( nop INTERFACE  ${CMAKE_SOURCE_DIR}/vendor/libnop/include )

add_subdirectory( asio )
add_executable( ex_pipeipe pipeipc/server.cpp )

add_executable( asio_server
    pipeipc/asio_server.cpp )
target_link_libraries( asio_server boost )

add_executable( asio_client
    pipeipc/asio_client.cpp )
target_link_libraries( asio_client boost )