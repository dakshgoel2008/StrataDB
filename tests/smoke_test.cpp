#include <gtest/gtest.h>

#include "strata/version.h"

TEST(Smoke, VersionIsNotEmpty) { EXPECT_FALSE(strata::version().empty()); }
