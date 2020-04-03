add_executable( client client.cpp )
target_link_libraries( client PUBLIC nop boost corobatch )
# target_compile_definitions( client PRIVATE COROBATCH_EXAMPLES_LOGGING_EXTRA )

add_executable( server server.cpp )
target_link_libraries( server PUBLIC nop boost )
# target_compile_definitions( server PRIVATE COROBATCH_EXAMPLES_LOGGING_EXTRA )
