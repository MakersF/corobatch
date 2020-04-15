add_executable( benchmark_multiply_add benchmark_multiply_add.cpp )
target_link_libraries( benchmark_multiply_add PRIVATE corobatch benchmark benchmark_main )
target_compile_definitions( benchmark_multiply_add PRIVATE COROBATCH_DISABLE_LOGGING )
target_compile_options( benchmark_multiply_add PRIVATE -mfma -fno-exceptions )
