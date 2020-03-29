add_library( common common.cpp )
target_include_directories( common PUBLIC comparision)

add_executable( nobatching nobatching.cpp )
target_include_directories( nobatching PUBLIC comparision)
target_link_libraries( nobatching PRIVATE common)

add_executable( handbatching handbatching.cpp )
target_include_directories( handbatching PUBLIC comparision)
target_link_libraries( handbatching PRIVATE common)

add_executable( corobatching corobatching.cpp )
target_include_directories( corobatching PUBLIC comparision )
target_link_libraries( corobatching PRIVATE corobatch common )