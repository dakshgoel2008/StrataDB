#include "strata/metric.h"

namespace strata {

std::string_view metric_name(Metric m) noexcept {
    switch (m) {
        case Metric::L2:
            return "l2";
        case Metric::InnerProduct:
            return "ip";
        case Metric::Cosine:
            return "cosine";
    }
    return "unknown";
}

std::optional<Metric> parse_metric(std::string_view name) noexcept {
    if (name == "l2") {
        return Metric::L2;
    }
    if (name == "ip" || name == "inner_product") {
        return Metric::InnerProduct;
    }
    if (name == "cosine") {
        return Metric::Cosine;
    }
    return std::nullopt;
}

}  // namespace strata
