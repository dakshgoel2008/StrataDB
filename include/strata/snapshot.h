#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "strata/hnsw_index.h"
#include "strata/metric.h"

namespace strata {

// The fixed-size header at the start of an HNSW snapshot file.
// docs/file-format.md describes every byte.
inline constexpr std::size_t kSnapshotHeaderBytes = 256;
inline constexpr std::uint32_t kSnapshotVersion = 1;

enum SnapshotSection : std::uint8_t {
    kQuantizerSection,
    kVectorsSection,
    kLevelsSection,
    kLayer0LinksSection,
    kUpperLinksSection,
    kSectionCount,
};

struct SnapshotHeader {
    struct Section {
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        std::uint32_t crc = 0;  // CRC32C of the section's bytes
    };

    Metric metric = Metric::L2;
    Encoding encoding = Encoding::Float;
    std::uint64_t dim = 0;
    std::uint64_t M = 0;
    std::uint64_t ef_construction = 0;
    std::uint64_t seed = 0;
    std::uint64_t capacity = 0;
    std::uint64_t count = 0;
    std::uint64_t entry = 0;      // entry node in the low 32 bits, its level in the high 32
    std::uint64_t wal_lsn = 0;    // last log record included (0 if none)
    std::uint64_t code_size = 0;  // bytes per compressed vector, 0 for full floats
    std::array<Section, kSectionCount> sections{};
};

std::array<std::uint8_t, kSnapshotHeaderBytes> encode_snapshot_header(const SnapshotHeader& h);

// Checks the magic bytes, version, header CRC, and that every field is in range
// and the sections have the sizes the other fields call for. The upper-links size
// can only be checked once the levels are read. Throws std::runtime_error.
SnapshotHeader parse_snapshot_header(std::span<const std::uint8_t> bytes);

}  // namespace strata
