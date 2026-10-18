// Prints the size of a .fvecs/.ivecs/.bvecs file and a few sample values.
// Usage: ./build/release/tools/vecs_info data/sift/base.fvecs

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

#include "bench/common/vecs_io.h"

namespace {

using strata::bench::Matrix;

constexpr std::size_t kSampleRows = 1000;
constexpr std::size_t kPrintValues = 8;

template <typename T>
void print_first_row(const Matrix<T>& m) {
    if (m.rows == 0) {
        return;
    }
    std::printf("row 0      : [");
    for (std::size_t j = 0; j < std::min(kPrintValues, m.dim); ++j) {
        std::printf("%s%g", j ? ", " : "", static_cast<double>(m.row(0)[j]));
    }
    std::printf("%s]\n", m.dim > kPrintValues ? ", ..." : "");
}

// Average vector length. Close to 1.0 means the vectors are already normalized.
void print_mean_length(const Matrix<float>& m) {
    if (m.rows == 0) {
        return;
    }
    double sum = 0;
    for (std::size_t i = 0; i < m.rows; ++i) {
        double squares = 0;
        for (std::size_t j = 0; j < m.dim; ++j) {
            const double v = m.row(i)[j];
            squares += v * v;
        }
        sum += std::sqrt(squares);
    }
    std::printf("mean length: %.4f (first %zu rows)\n", sum / static_cast<double>(m.rows), m.rows);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <file.fvecs|file.ivecs|file.bvecs>\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const std::string ext = std::filesystem::path(path).extension().string();

    try {
        namespace sb = strata::bench;
        if (ext == ".fvecs") {
            const sb::VecsShape shape = sb::inspect_vecs(path, sizeof(float));
            std::printf("rows x dim : %zu x %zu (float32)\n", shape.rows, shape.dim);
            const auto sample = sb::read_fvecs(path, kSampleRows);
            print_first_row(sample);
            print_mean_length(sample);
        } else if (ext == ".ivecs") {
            const sb::VecsShape shape = sb::inspect_vecs(path, sizeof(std::int32_t));
            std::printf("rows x dim : %zu x %zu (int32)\n", shape.rows, shape.dim);
            print_first_row(sb::read_ivecs(path, kSampleRows));
        } else if (ext == ".bvecs") {
            const sb::VecsShape shape = sb::inspect_vecs(path, sizeof(std::uint8_t));
            std::printf("rows x dim : %zu x %zu (uint8)\n", shape.rows, shape.dim);
            const auto sample = sb::read_bvecs(path, kSampleRows);
            print_first_row(sample);
            print_mean_length(sample);
        } else {
            std::fprintf(stderr, "unknown extension '%s'\n", ext.c_str());
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
