find_package(Threads REQUIRED)

add_library(strata
  version.cpp
  metric.cpp
  distance.cpp
  flat_index.cpp
  hnsw_index.cpp
)

target_include_directories(strata PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(strata PRIVATE STRATA_VERSION="${PROJECT_VERSION}")
target_link_libraries(strata PUBLIC Threads::Threads PRIVATE strata_warnings)
