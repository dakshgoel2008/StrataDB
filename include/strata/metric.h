#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace strata {

// How closeness between two vectors is measured.
// Every metric is turned into a distance where smaller means closer.
enum class Metric : std::uint8_t {
    L2,            // squared straight-line distance
    InnerProduct,  // distance = -dot(a, b)
    Cosine,        // distance = 1 - cosine similarity; vectors are scaled to length 1 first
};

std::string_view metric_name(Metric m) noexcept;

// Accepts "l2", "ip" / "inner_product", "cosine". Returns nothing for unknown names.
std::optional<Metric> parse_metric(std::string_view name) noexcept;

}  // namespace strata
