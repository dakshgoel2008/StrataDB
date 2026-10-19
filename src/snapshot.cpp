#include "strata/snapshot.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "checksum_policy.h"
#include "hnsw_entry.h"
#include "strata/crc32c.h"
#include "strata/hnsw_index.h"

namespace strata {
namespace {

constexpr char kMagic[8] = {'S', 'T', 'R', 'S', 'N', 'A', 'P', '1'};
constexpr std::size_t kSectionTableOffset = 96;
constexpr std::size_t kSectionEntryBytes = 24;
constexpr std::size_t kCrcOffset = kSnapshotHeaderBytes - 4;
constexpr std::size_t kIoChunk = 1u << 20;

template <typename T>
void put(std::uint8_t* out, std::size_t offset, T value) {
    std::memcpy(out + offset, &value, sizeof(T));
}

template <typename T>
T get(const std::uint8_t* in, std::size_t offset) {
    T value;
    std::memcpy(&value, in + offset, sizeof(T));
    return value;
}

[[noreturn]] void bad(const std::string& why) {
    throw std::runtime_error("snapshot is damaged: " + why);
}

// Streams one section after another into the file, keeping each one's CRC.
class SectionWriter {
public:
    explicit SectionWriter(WritableFile& file) : file_(file) { buffer_.reserve(kIoChunk); }

    void put_bytes(const void* data, std::size_t size) {
        crc_ = crc32c(data, size, crc_);
        size_ += size;
        const auto* p = static_cast<const std::uint8_t*>(data);
        buffer_.insert(buffer_.end(), p, p + size);
        if (buffer_.size() >= kIoChunk) {
            flush();
        }
    }

    SnapshotHeader::Section finish() {
        flush();
        const SnapshotHeader::Section s{offset_, size_, crc_};
        offset_ += size_;
        size_ = 0;
        crc_ = 0;
        return s;
    }

private:
    void flush() {
        file_.append(buffer_.data(), buffer_.size());
        buffer_.clear();
    }

    WritableFile& file_;
    std::vector<std::uint8_t> buffer_;
    std::uint64_t offset_ = kSnapshotHeaderBytes;
    std::uint64_t size_ = 0;
    std::uint32_t crc_ = 0;
};

// Reads one section in large chunks and checks its CRC at the end.
class SectionReader {
public:
    SectionReader(const ReadableFile& file, const SnapshotHeader::Section& section,
                  const char* name)
        : file_(file), section_(section), name_(name), pos_(section.offset) {}

    void get_bytes(void* out, std::size_t size) {
        auto* p = static_cast<std::uint8_t*>(out);
        while (size > 0) {
            if (used_ == buffer_.size()) {
                refill();
            }
            const std::size_t n = std::min(size, buffer_.size() - used_);
            std::memcpy(p, buffer_.data() + used_, n);
            used_ += n;
            p += n;
            size -= n;
        }
    }

    void finish() {
        if (used_ != buffer_.size() || pos_ != section_.offset + section_.size) {
            bad(std::string(name_) + " section has the wrong size");
        }
        if (kVerifyChecksums && crc_ != section_.crc) {
            bad(std::string(name_) + " section fails its checksum");
        }
    }

private:
    void refill() {
        const std::uint64_t left = section_.offset + section_.size - pos_;
        if (left == 0) {
            bad(std::string(name_) + " section is shorter than expected");
        }
        buffer_.resize(static_cast<std::size_t>(std::min<std::uint64_t>(left, kIoChunk)));
        file_.read(pos_, buffer_.data(), buffer_.size());
        crc_ = crc32c(buffer_.data(), buffer_.size(), crc_);
        pos_ += buffer_.size();
        used_ = 0;
    }

    const ReadableFile& file_;
    SnapshotHeader::Section section_;
    const char* name_;
    std::uint64_t pos_;
    std::vector<std::uint8_t> buffer_;
    std::size_t used_ = 0;
    std::uint32_t crc_ = 0;
};

}  // namespace

std::array<std::uint8_t, kSnapshotHeaderBytes> encode_snapshot_header(const SnapshotHeader& h) {
    std::array<std::uint8_t, kSnapshotHeaderBytes> out{};
    std::uint8_t* p = out.data();
    std::memcpy(p, kMagic, sizeof(kMagic));
    put<std::uint32_t>(p, 8, kSnapshotVersion);
    put<std::uint32_t>(p, 12, static_cast<std::uint32_t>(kSnapshotHeaderBytes));
    put<std::uint8_t>(p, 16, static_cast<std::uint8_t>(h.metric));
    put<std::uint8_t>(p, 17, static_cast<std::uint8_t>(h.encoding));
    put<std::uint64_t>(p, 24, h.dim);
    put<std::uint64_t>(p, 32, h.M);
    put<std::uint64_t>(p, 40, h.ef_construction);
    put<std::uint64_t>(p, 48, h.seed);
    put<std::uint64_t>(p, 56, h.capacity);
    put<std::uint64_t>(p, 64, h.count);
    put<std::uint64_t>(p, 72, h.entry);
    put<std::uint64_t>(p, 80, h.wal_lsn);
    put<std::uint64_t>(p, 88, h.code_size);
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        const std::size_t at = kSectionTableOffset + i * kSectionEntryBytes;
        put<std::uint64_t>(p, at, h.sections[i].offset);
        put<std::uint64_t>(p, at + 8, h.sections[i].size);
        put<std::uint32_t>(p, at + 16, h.sections[i].crc);
    }
    put<std::uint32_t>(p, kCrcOffset, crc32c(p, kCrcOffset));
    return out;
}

SnapshotHeader parse_snapshot_header(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < kSnapshotHeaderBytes) {
        bad("header is cut off");
    }
    const std::uint8_t* p = bytes.data();
    if (std::memcmp(p, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("not a StrataDB snapshot file");
    }
    if (get<std::uint32_t>(p, 8) != kSnapshotVersion ||
        get<std::uint32_t>(p, 12) != kSnapshotHeaderBytes) {
        throw std::runtime_error("unsupported snapshot version");
    }
    if (kVerifyChecksums && get<std::uint32_t>(p, kCrcOffset) != crc32c(p, kCrcOffset)) {
        bad("header fails its checksum");
    }

    SnapshotHeader h;
    const auto metric = get<std::uint8_t>(p, 16);
    const auto encoding = get<std::uint8_t>(p, 17);
    if (metric > static_cast<std::uint8_t>(Metric::Cosine) ||
        encoding > static_cast<std::uint8_t>(Encoding::PQ)) {
        bad("unknown metric or encoding");
    }
    h.metric = static_cast<Metric>(metric);
    h.encoding = static_cast<Encoding>(encoding);
    h.dim = get<std::uint64_t>(p, 24);
    h.M = get<std::uint64_t>(p, 32);
    h.ef_construction = get<std::uint64_t>(p, 40);
    h.seed = get<std::uint64_t>(p, 48);
    h.capacity = get<std::uint64_t>(p, 56);
    h.count = get<std::uint64_t>(p, 64);
    h.entry = get<std::uint64_t>(p, 72);
    h.wal_lsn = get<std::uint64_t>(p, 80);
    h.code_size = get<std::uint64_t>(p, 88);
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        const std::size_t at = kSectionTableOffset + i * kSectionEntryBytes;
        h.sections[i].offset = get<std::uint64_t>(p, at);
        h.sections[i].size = get<std::uint64_t>(p, at + 8);
        h.sections[i].crc = get<std::uint32_t>(p, at + 16);
    }

    if (h.dim == 0 || h.dim > 65536 || h.M < 2 || h.M > 1024 || h.ef_construction == 0) {
        bad("dim or graph settings out of range");
    }
    if (h.capacity == 0 || h.capacity >= std::numeric_limits<std::uint32_t>::max() ||
        h.count > h.capacity) {
        bad("count or capacity out of range");
    }
    std::uint64_t vector_bytes = 0;
    std::uint64_t quantizer_bytes = 0;
    switch (h.encoding) {
        case Encoding::Float:
            if (h.code_size != 0) {
                bad("full-float snapshot with a code size");
            }
            vector_bytes = h.dim * sizeof(float);
            break;
        case Encoding::SQ8:
            if (h.code_size != h.dim) {
                bad("SQ8 code size must equal dim");
            }
            vector_bytes = h.dim;
            quantizer_bytes = 2 * h.dim * sizeof(float);
            break;
        case Encoding::PQ:
            if (h.code_size == 0 || h.dim % h.code_size != 0) {
                bad("PQ piece count must divide dim");
            }
            vector_bytes = h.code_size;
            quantizer_bytes = ProductQuantizer::kCentroids * h.dim * sizeof(float);
            break;
    }

    // Sections follow the header back to back, in order.
    const std::uint64_t expected[kSectionCount] = {
        quantizer_bytes,
        h.count * vector_bytes,
        h.count,
        h.count * (1 + 2 * h.M) * sizeof(std::uint32_t),
        h.sections[kUpperLinksSection].size,
    };
    std::uint64_t offset = kSnapshotHeaderBytes;
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        if (h.sections[i].offset != offset || h.sections[i].size != expected[i]) {
            bad("section table doesn't match the header");
        }
        offset += h.sections[i].size;
    }
    if (h.sections[kUpperLinksSection].size % ((1 + h.M) * sizeof(std::uint32_t)) != 0 ||
        h.sections[kUpperLinksSection].size >
            h.count * hnsw::kMaxLevel * (1 + h.M) * sizeof(std::uint32_t)) {
        bad("upper links section has an impossible size");
    }
    if (h.count > 0) {
        const auto node = hnsw::entry_node(h.entry);
        const auto level = hnsw::entry_level(h.entry);
        if (node >= h.count || level < 0 || level > hnsw::kMaxLevel) {
            bad("entry point out of range");
        }
    }
    return h;
}

HnswIndex::SavePoint HnswIndex::save_point() const {
    return {size(), entry_.load(std::memory_order_acquire)};
}

void HnswIndex::save_snapshot(FileSystem& fs, const std::string& path, const SavePoint& point,
                              std::uint64_t wal_lsn) const {
    const std::size_t n = point.count;
    if (n > size()) {
        throw std::invalid_argument("save point is newer than the index");
    }
    SnapshotHeader h;
    h.metric = metric_;
    h.encoding = encoding_;
    h.dim = dim_;
    h.M = params_.M;
    h.ef_construction = params_.ef_construction;
    h.seed = params_.seed;
    h.capacity = capacity_;
    h.count = n;
    h.entry = n > 0 ? point.entry : 0;
    h.wal_lsn = wal_lsn;
    h.code_size = encoding_ == Encoding::Float ? 0 : code_size_;

    const std::string temp = path + ".tmp";
    {
        const auto file = fs.create(temp);
        const std::array<std::uint8_t, kSnapshotHeaderBytes> blank{};
        file->append(blank.data(), blank.size());  // the real header goes here at the end
        SectionWriter out(*file);

        if (encoding_ == Encoding::SQ8) {
            out.put_bytes(sq_->mins().data(), dim_ * sizeof(float));
            out.put_bytes(sq_->scales().data(), dim_ * sizeof(float));
        } else if (encoding_ == Encoding::PQ) {
            const auto& c = pq_->all_centroids();
            out.put_bytes(c.data(), c.size() * sizeof(float));
        }
        h.sections[kQuantizerSection] = out.finish();

        for (NodeId node = 0; node < n; ++node) {
            if (encoding_ == Encoding::Float) {
                out.put_bytes(vector_of(node), dim_ * sizeof(float));
            } else {
                out.put_bytes(code_of(node), code_size_);
            }
        }
        h.sections[kVectorsSection] = out.finish();

        out.put_bytes(levels_.data(), n);
        h.sections[kLevelsSection] = out.finish();

        const auto limit = static_cast<NodeId>(n);
        std::vector<NodeId> block;
        for (NodeId node = 0; node < n; ++node) {
            copy_block(node, 0, limit, block);
            out.put_bytes(block.data(), block.size() * sizeof(NodeId));
        }
        h.sections[kLayer0LinksSection] = out.finish();

        for (NodeId node = 0; node < n; ++node) {
            for (int l = 1; l <= levels_[node]; ++l) {
                copy_block(node, l, limit, block);
                out.put_bytes(block.data(), block.size() * sizeof(NodeId));
            }
        }
        h.sections[kUpperLinksSection] = out.finish();

        const auto header = encode_snapshot_header(h);
        file->write_at(0, header.data(), header.size());
        file->sync();
    }
    fs.rename(temp, path);
    fs.sync_dir(parent_dir(path));
}

HnswIndex::LoadedSnapshot HnswIndex::load_snapshot(FileSystem& fs, const std::string& path,
                                                   std::size_t capacity) {
    const auto file = fs.open_read(path);
    std::array<std::uint8_t, kSnapshotHeaderBytes> raw{};
    if (file->size() < raw.size()) {
        throw std::runtime_error("'" + path + "' is too small to be a snapshot");
    }
    file->read(0, raw.data(), raw.size());
    const SnapshotHeader h = parse_snapshot_header(raw);
    const auto& last = h.sections[kUpperLinksSection];
    if (last.offset + last.size != file->size()) {
        bad("file size doesn't match the header");
    }

    const auto dim = static_cast<std::size_t>(h.dim);
    const auto n = static_cast<std::size_t>(h.count);
    const HnswParams params{static_cast<std::size_t>(h.M),
                            static_cast<std::size_t>(h.ef_construction), h.seed};
    const std::size_t cap =
        capacity == 0 ? static_cast<std::size_t>(h.capacity) : std::max(capacity, n);
    if (cap == 0 || cap >= std::numeric_limits<NodeId>::max()) {
        throw std::invalid_argument("capacity must be between 1 and 2^32 - 2");
    }

    std::unique_ptr<HnswIndex> index;
    {
        SectionReader in(*file, h.sections[kQuantizerSection], "quantizer");
        if (h.encoding == Encoding::SQ8) {
            std::vector<float> mins(dim);
            std::vector<float> scales(dim);
            in.get_bytes(mins.data(), dim * sizeof(float));
            in.get_bytes(scales.data(), dim * sizeof(float));
            in.finish();
            index = std::make_unique<HnswIndex>(
                dim, h.metric, cap, params,
                ScalarQuantizer::from_ranges(std::move(mins), std::move(scales)));
        } else if (h.encoding == Encoding::PQ) {
            std::vector<float> centroids(ProductQuantizer::kCentroids * dim);
            in.get_bytes(centroids.data(), centroids.size() * sizeof(float));
            in.finish();
            index = std::make_unique<HnswIndex>(
                dim, h.metric, cap, params,
                ProductQuantizer::from_centroids(dim, static_cast<std::size_t>(h.code_size),
                                                 std::move(centroids)));
        } else {
            in.finish();
            index = std::make_unique<HnswIndex>(dim, h.metric, cap, params);
        }
    }

    {
        SectionReader in(*file, h.sections[kVectorsSection], "vectors");
        for (NodeId node = 0; node < n; ++node) {
            if (h.encoding == Encoding::Float) {
                in.get_bytes(index->vector_of(node), dim * sizeof(float));
            } else {
                in.get_bytes(index->code_of(node), index->code_size_);
            }
        }
        in.finish();
    }

    std::size_t upper_blocks = 0;
    {
        SectionReader in(*file, h.sections[kLevelsSection], "levels");
        in.get_bytes(index->levels_.data(), n);
        in.finish();
        for (NodeId node = 0; node < n; ++node) {
            if (index->levels_[node] > hnsw::kMaxLevel) {
                bad("node level out of range");
            }
            upper_blocks += index->levels_[node];
        }
    }
    if (upper_blocks * (1 + params.M) * sizeof(NodeId) != h.sections[kUpperLinksSection].size) {
        bad("upper links section doesn't match the node levels");
    }

    {
        SectionReader in(*file, h.sections[kLayer0LinksSection], "layer-0 links");
        in.get_bytes(index->level0_links_.data(),
                     n * (1 + index->max_links_level0_) * sizeof(NodeId));
        in.finish();
    }
    {
        SectionReader in(*file, h.sections[kUpperLinksSection], "upper links");
        for (NodeId node = 0; node < n; ++node) {
            const std::size_t level = index->levels_[node];
            if (level > 0) {
                auto& block = index->upper_links_[node];
                block.resize(level * (1 + params.M));
                in.get_bytes(block.data(), block.size() * sizeof(NodeId));
            }
        }
        in.finish();
    }

    // Checksums catch damage; this also stops a file that was written wrongly
    // from making a search read out of bounds.
    index->check_links(n);
    if (n > 0 && hnsw::entry_level(h.entry) != index->levels_[hnsw::entry_node(h.entry)]) {
        bad("entry point level doesn't match the node");
    }
    index->count_.store(n);
    index->entry_.store(n > 0 ? h.entry : hnsw::kNoEntry);
    return {std::move(index), h.wal_lsn};
}

void HnswIndex::save(const std::string& path) const {
    save_snapshot(local_file_system(), path, save_point(), 0);
}

std::unique_ptr<HnswIndex> HnswIndex::load(const std::string& path, std::size_t capacity) {
    return load_snapshot(local_file_system(), path, capacity).index;
}

}  // namespace strata
