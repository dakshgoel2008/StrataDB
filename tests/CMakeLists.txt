add_executable(strata_tests
  smoke_test.cpp
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
