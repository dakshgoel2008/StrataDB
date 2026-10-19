#pragma once

namespace strata {

// Fuzzing builds skip checksum checks, so the fuzzer can reach the code that runs
// after them. Every other build checks them.
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
inline constexpr bool kVerifyChecksums = false;
#else
inline constexpr bool kVerifyChecksums = true;
#endif

}  // namespace strata
