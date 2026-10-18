add_library(strata_bench_common
  common/vecs_io.cpp
  common/metrics.cpp
)

# Lets code write: #include "bench/common/vecs_io.h"
target_include_directories(strata_bench_common PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(strata_bench_common PRIVATE strata_warnings)
