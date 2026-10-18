#pragma once

#include <string_view>

namespace strata {

// Library version, e.g. "0.1.0" (set in CMakeLists.txt).
std::string_view version() noexcept;

}  // namespace strata
