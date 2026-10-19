# libFuzzer targets. Build with clang using the "fuzz" preset.
foreach(target wal_fuzz snapshot_fuzz)
  add_executable(${target} ${target}.cpp)
  target_link_libraries(${target} PRIVATE strata strata_warnings)
  target_link_options(${target} PRIVATE -fsanitize=fuzzer)
endforeach()

# Writes valid files for the fuzzers to start from.
add_executable(make_seeds make_seeds.cpp)
target_link_libraries(make_seeds PRIVATE strata strata_warnings)
