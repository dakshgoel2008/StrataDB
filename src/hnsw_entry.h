#pragma once

#include <cstdint>

// Helpers shared by the HNSW code and its snapshot code.
namespace strata::hnsw {

inline constexpr std::uint64_t kNoEntry = ~std::uint64_t{0};
inline constexpr int kMaxLevel = 30;

// The entry point is one node id and its level packed into 64 bits.
inline std::uint64_t pack_entry(std::uint32_t node, int level) {
    return (static_cast<std::uint64_t>(level) << 32) | node;
}

inline std::uint32_t entry_node(std::uint64_t entry) { return static_cast<std::uint32_t>(entry); }

inline int entry_level(std::uint64_t entry) { return static_cast<int>(entry >> 32); }

}  // namespace strata::hnsw
