#pragma once

#include <cstddef>
#include <cstdint>

namespace strata {

// CRC32C checksum (the Castagnoli polynomial, as used by ext4, iSCSI and RocksDB).
// To checksum data in pieces, pass the previous result as `crc`.
// Uses the CPU's crc32 instruction when it has one.
std::uint32_t crc32c(const void* data, std::size_t size, std::uint32_t crc = 0) noexcept;

// The plain table version, always available. Exposed so tests can compare both.
std::uint32_t crc32c_portable(const void* data, std::size_t size, std::uint32_t crc = 0) noexcept;

// True if crc32c() is using the CPU instruction.
bool crc32c_uses_cpu_instruction() noexcept;

}  // namespace strata
