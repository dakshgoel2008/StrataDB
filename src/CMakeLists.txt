add_library(strata
  version.cpp
)

target_include_directories(strata PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(strata PRIVATE STRATA_VERSION="${PROJECT_VERSION}")
target_link_libraries(strata PRIVATE strata_warnings)
