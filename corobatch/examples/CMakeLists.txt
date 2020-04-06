add_library( boost INTERFACE )
target_include_directories( boost INTERFACE  ${CMAKE_SOURCE_DIR}/vendor/boost )

add_library( nop INTERFACE )
target_include_directories( nop INTERFACE  ${CMAKE_SOURCE_DIR}/vendor/libnop/include )

add_subdirectory( asio )
