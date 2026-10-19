// Loads random bytes as a snapshot file. Loading may refuse the file, but must
// never crash, read out of bounds, or leave an index that crashes a search.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <vector>

#include "strata/file_system.h"
#include "strata/hnsw_index.h"
#include "strata/snapshot.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    strata::MemoryFileSystem fs;
    fs.make_dir("/d");
    {
        auto file = fs.create("/d/s.snap");
        file->append(data, size);
    }
    try {
        // Capacity 1 means "just big enough for the saved rows", so a fake huge
        // capacity in the header can't make the fuzzer run out of memory.
        auto loaded = strata::HnswIndex::load_snapshot(fs, "/d/s.snap", 1);
        const auto& index = *loaded.index;
        if (index.size() > 0) {
            std::vector<float> query(index.dim(), 0.5f);
            std::vector<strata::RowId> ids(3);
            std::vector<float> dist(3);
            index.search(query, 3, 10, ids, dist);
        }
    } catch (const std::exception&) {
        return 0;  // refusing a bad file is fine
    }
    return 0;
}
