#include "bench/common/vecs_io.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace strata::bench {
namespace {

// Closes the file automatically, even if an error is thrown.
struct FileCloser {
    void operator()(std::FILE* f) const noexcept { std::fclose(f); }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

FilePtr open_or_throw(const std::string& path, const char* mode) {
    FilePtr f(std::fopen(path.c_str(), mode));
    if (!f) {
        throw std::runtime_error("cannot open '" + path + "'");
    }
    return f;
}

// fread can return fewer bytes than asked, so always check the count.
void read_exact(std::FILE* f, void* dst, std::size_t bytes, const std::string& path) {
    if (std::fread(dst, 1, bytes, f) != bytes) {
        throw std::runtime_error("unexpected end of file in '" + path + "'");
    }
}

void write_exact(std::FILE* f, const void* src, std::size_t bytes, const std::string& path) {
    if (std::fwrite(src, 1, bytes, f) != bytes) {
        throw std::runtime_error("write failed for '" + path + "'");
    }
}

// FileType is what's stored on disk, OutType is what we return.
template <typename FileType, typename OutType>
Matrix<OutType> read_vecs(const std::string& path, std::size_t max_rows, std::size_t first_row) {
    const VecsShape shape = inspect_vecs(path, sizeof(FileType));
    const std::size_t available = shape.rows - std::min(first_row, shape.rows);
    const std::size_t rows = (max_rows == 0) ? available : std::min(max_rows, available);

    Matrix<OutType> m;
    m.rows = rows;
    m.dim = shape.dim;
    m.data.resize(rows * shape.dim);

    FilePtr f = open_or_throw(path, "rb");
    if (rows > 0 && first_row > 0) {
        const std::size_t row_bytes = sizeof(std::int32_t) + shape.dim * sizeof(FileType);
        if (std::fseek(f.get(), static_cast<long>(first_row * row_bytes), SEEK_SET) != 0) {
            throw std::runtime_error("seek failed in '" + path + "'");
        }
    }
    std::vector<FileType> row_buffer;  // only needed when the types differ (bvecs)
    if constexpr (!std::is_same_v<FileType, OutType>) {
        row_buffer.resize(shape.dim);
    }

    for (std::size_t i = 0; i < rows; ++i) {
        std::int32_t d = 0;
        read_exact(f.get(), &d, sizeof(d), path);
        // Every row must have the same dim as the first one.
        if (d < 0 || static_cast<std::size_t>(d) != shape.dim) {
            throw std::runtime_error("row " + std::to_string(first_row + i) + " of '" + path +
                                     "' has dim " + std::to_string(d) + ", expected " +
                                     std::to_string(shape.dim));
        }
        if constexpr (std::is_same_v<FileType, OutType>) {
            read_exact(f.get(), m.row(i), shape.dim * sizeof(FileType), path);
        } else {
            read_exact(f.get(), row_buffer.data(), shape.dim * sizeof(FileType), path);
            std::transform(row_buffer.begin(), row_buffer.end(), m.row(i),
                           [](FileType v) { return static_cast<OutType>(v); });
        }
    }
    return m;
}

template <typename T>
void write_vecs(const std::string& path, const Matrix<T>& m) {
    if (m.dim == 0 || m.dim > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("invalid dim for '" + path + "'");
    }
    if (m.data.size() != m.rows * m.dim) {
        throw std::runtime_error("matrix data size does not match rows * dim");
    }

    FilePtr f = open_or_throw(path, "wb");
    const auto d = static_cast<std::int32_t>(m.dim);
    for (std::size_t i = 0; i < m.rows; ++i) {
        write_exact(f.get(), &d, sizeof(d), path);
        write_exact(f.get(), m.row(i), m.dim * sizeof(T), path);
    }
    // A full disk is often only reported when the file is closed.
    if (std::fclose(f.release()) != 0) {
        throw std::runtime_error("close failed for '" + path + "'");
    }
}

}  // namespace

VecsShape inspect_vecs(const std::string& path, std::size_t value_bytes) {
    std::error_code ec;
    const std::uintmax_t file_size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("cannot read size of '" + path + "': " + ec.message());
    }
    if (file_size == 0) {
        return {};
    }

    FilePtr f = open_or_throw(path, "rb");
    std::int32_t d = 0;
    read_exact(f.get(), &d, sizeof(d), path);
    if (d <= 0) {
        throw std::runtime_error("invalid dim " + std::to_string(d) + " in '" + path + "'");
    }

    // A file cut off midway won't divide evenly into rows.
    const std::size_t dim = static_cast<std::size_t>(d);
    const std::uintmax_t row_bytes = sizeof(std::int32_t) + dim * value_bytes;
    if (file_size % row_bytes != 0) {
        throw std::runtime_error("size of '" + path + "' doesn't match its row size " +
                                 "(cut-off file or wrong format?)");
    }
    return {static_cast<std::size_t>(file_size / row_bytes), dim};
}

Matrix<float> read_fvecs(const std::string& path, std::size_t max_rows, std::size_t first_row) {
    return read_vecs<float, float>(path, max_rows, first_row);
}

Matrix<std::int32_t> read_ivecs(const std::string& path, std::size_t max_rows,
                                std::size_t first_row) {
    return read_vecs<std::int32_t, std::int32_t>(path, max_rows, first_row);
}

Matrix<float> read_bvecs(const std::string& path, std::size_t max_rows, std::size_t first_row) {
    return read_vecs<std::uint8_t, float>(path, max_rows, first_row);
}

void write_fvecs(const std::string& path, const Matrix<float>& m) { write_vecs(path, m); }

void write_ivecs(const std::string& path, const Matrix<std::int32_t>& m) { write_vecs(path, m); }

}  // namespace strata::bench
