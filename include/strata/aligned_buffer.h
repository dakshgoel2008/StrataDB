#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>

namespace strata {

// A growable array whose start is on a 64-byte boundary (one CPU cache line).
// Fast vector instructions in later phases work best on memory aligned like this.
template <typename T>
class AlignedBuffer {
    static_assert(std::is_trivially_copyable_v<T>, "AlignedBuffer only holds plain values");

public:
    static constexpr std::size_t kAlignment = 64;

    AlignedBuffer() = default;
    AlignedBuffer(AlignedBuffer&&) noexcept = default;
    AlignedBuffer& operator=(AlignedBuffer&&) noexcept = default;

    T* data() noexcept { return data_.get(); }
    const T* data() const noexcept { return data_.get(); }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }

    // New elements are set to zero.
    void resize(std::size_t n) {
        if (n > capacity_) {
            reserve(std::max(n, capacity_ * 2));
        }
        if (n > size_) {
            std::memset(data_.get() + size_, 0, (n - size_) * sizeof(T));
        }
        size_ = n;
    }

    void reserve(std::size_t n) {
        if (n <= capacity_) {
            return;
        }
        // aligned_alloc needs the byte size to be a multiple of the alignment.
        std::size_t bytes = n * sizeof(T);
        bytes = (bytes + kAlignment - 1) / kAlignment * kAlignment;
        auto* fresh = static_cast<T*>(std::aligned_alloc(kAlignment, bytes));
        if (fresh == nullptr) {
            throw std::bad_alloc();
        }
        if (size_ > 0) {
            std::memcpy(fresh, data_.get(), size_ * sizeof(T));
        }
        data_.reset(fresh);
        capacity_ = n;
    }

private:
    struct Free {
        void operator()(T* p) const noexcept { std::free(p); }
    };

    std::unique_ptr<T[], Free> data_;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

}  // namespace strata
