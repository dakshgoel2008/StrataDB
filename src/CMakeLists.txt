find_package(Threads REQUIRED)

add_library(strata
  version.cpp
  metric.cpp
  distance.cpp
  flat_index.cpp
  hnsw_index.cpp
  quantizer.cpp
  crc32c.cpp
  file_system.cpp
  memory_file_system.cpp
  wal.cpp
  snapshot.cpp
  database.cpp
)

# Fast distance kernels for x86. Each file gets its own CPU flags, and the
# program checks at runtime which ones the CPU can actually run.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  target_sources(strata PRIVATE distance_autovec.cpp distance_avx2.cpp distance_avx512.cpp
                                crc32c_sse42.cpp)
  set_source_files_properties(distance_autovec.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffast-math")
  set_source_files_properties(distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  set_source_files_properties(distance_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f")
  set_source_files_properties(crc32c_sse42.cpp PROPERTIES COMPILE_OPTIONS "-msse4.2")
  target_compile_definitions(strata PRIVATE STRATA_X86_KERNELS)
endif()

target_include_directories(strata PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(strata PRIVATE STRATA_VERSION="${PROJECT_VERSION}")
target_link_libraries(strata PUBLIC Threads::Threads PRIVATE strata_warnings)
