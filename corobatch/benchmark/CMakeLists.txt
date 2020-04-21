option(COROBATCH_BENCHMARK "Compile the benchmarks" ON)

if(COROBATCH_BENCHMARK)
    add_library( benchmark INTERFACE )
    target_link_libraries( benchmark INTERFACE ${CMAKE_SOURCE_DIR}/vendor/benchmark/build/src/libbenchmark.a )
    target_include_directories( benchmark INTERFACE ${CMAKE_SOURCE_DIR}/vendor/benchmark/include )

    add_library( benchmark_main STATIC IMPORTED )
    set_target_properties( benchmark_main PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/vendor/benchmark/build/src/libbenchmark_main.a )

    add_subdirectory( multiply_add )
    add_subdirectory( remote_call )
endif()
