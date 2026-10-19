add_executable(strata_tests
  smoke_test.cpp
  aligned_buffer_test.cpp
  distance_test.cpp
  flat_index_test.cpp
  hnsw_index_test.cpp
  quantizer_test.cpp
  crc32c_test.cpp
  file_system_test.cpp
  wal_test.cpp
  snapshot_test.cpp
  database_test.cpp
  bench_common/vecs_io_test.cpp
  bench_common/metrics_test.cpp
)

target_link_libraries(strata_tests PRIVATE
  strata
  strata_bench_common
  strata_warnings
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(strata_tests)
