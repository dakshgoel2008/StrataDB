#pragma once

// Reads and writes the vector files used by standard search benchmarks.
// Each row in the file is: [int32 dim][dim values]. No file header.
//   .fvecs = float values, .ivecs = int values, .bvecs = byte values

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::bench {

// A table of numbers stored in one flat array. Row i starts at data[i * dim].
template <typename T>
struct Matrix {
    std::size_t rows = 0;
    std::size_t dim = 0;
    std::vector<T> data;

    T* row(std::size_t i) { return data.data() + i * dim; }
    const T* row(std::size_t i) const { return data.data() + i * dim; }
};

struct VecsShape {
    std::size_t rows = 0;
    std::size_t dim = 0;
};

// Gets the row count and dim without reading the whole file.
// value_bytes is 4 for fvecs/ivecs and 1 for bvecs.
VecsShape inspect_vecs(const std::string& path, std::size_t value_bytes);

// Reads up to max_rows rows (0 means all). Throws if the file is broken.
Matrix<float> read_fvecs(const std::string& path, std::size_t max_rows = 0);
Matrix<std::int32_t> read_ivecs(const std::string& path, std::size_t max_rows = 0);
Matrix<float> read_bvecs(const std::string& path, std::size_t max_rows = 0);  // bytes become floats

void write_fvecs(const std::string& path, const Matrix<float>& m);
void write_ivecs(const std::string& path, const Matrix<std::int32_t>& m);

}  // namespace strata::bench
