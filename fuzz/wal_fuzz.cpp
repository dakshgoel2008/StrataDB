// Feeds random bytes to the log reader, then opens a database whose log file
// holds those bytes. Opening may refuse the data, but must never crash or
// read out of bounds.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>

#include "strata/database.h"
#include "strata/wal.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::span<const std::uint8_t> bytes(data, size);
    strata::read_wal_records(bytes, 1, [](const strata::WalRecord& r) {
        strata::InsertRecord ins;
        strata::decode_insert(r.payload, ins);
    });

    strata::MemoryFileSystem fs;
    strata::DatabaseOptions options;
    options.fs = &fs;
    options.replay_threads = 1;
    strata::Database::create("/db",
                             std::make_unique<strata::HnswIndex>(4, strata::Metric::L2, 64,
                                                                 strata::HnswParams{4, 16, 1}),
                             options);
    {
        auto log = fs.create("/db/" + strata::wal_file_name(1));
        log->append(data, size);
    }
    try {
        auto db = strata::Database::open("/db", options);
        db->insert(std::vector<float>(4, 1.0f));
    } catch (const std::exception&) {
        return 0;  // refusing bad data is fine
    }
    return 0;
}
