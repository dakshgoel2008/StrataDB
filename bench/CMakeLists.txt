add_library(strata_bench_common
  common/vecs_io.cpp
  common/metrics.cpp
)

# Lets code write: #include "bench/common/vecs_io.h"
target_include_directories(strata_bench_common PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(strata_bench_common PRIVATE strata_warnings)

# hnswlib is only used here, to compare against. The engine never uses it.
# SOURCE_SUBDIR points nowhere so CMake only downloads it and builds nothing.
include(FetchContent)
FetchContent_Declare(hnswlib
  URL https://github.com/nmslib/hnswlib/archive/refs/tags/v0.8.0.tar.gz
  URL_HASH SHA256=cf61d7dd8dc9bfba7f4abe0ed26698e90ac9f49a0badb2e1b0f3ba89b72cf3bb
  SOURCE_SUBDIR do-not-build)
FetchContent_MakeAvailable(hnswlib)
add_library(hnswlib_headers INTERFACE)
target_include_directories(hnswlib_headers SYSTEM INTERFACE ${hnswlib_SOURCE_DIR})

add_executable(hnsw_bench hnsw_bench.cpp)
target_link_libraries(hnsw_bench PRIVATE strata strata_bench_common hnswlib_headers strata_warnings)
# Lets hnswlib use this CPU's fastest instructions (AVX2), as its Python package does.
# StrataDB's own code is in libstrata and is not affected.
target_compile_options(hnsw_bench PRIVATE -march=native)

FetchContent_Declare(googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz
  URL_HASH SHA256=9631341c82bac4a288bef951f8b26b41f69021794184ece969f8473977eaa340
  SYSTEM)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(distance_bench distance_bench.cpp)
target_link_libraries(distance_bench PRIVATE strata benchmark::benchmark strata_warnings)

add_executable(durability_bench durability_bench.cpp)
target_link_libraries(durability_bench PRIVATE strata strata_bench_common strata_warnings)
