#ifndef INTERNAL_UTILS_H
#define INTERNAL_UTILS_H

#include <vector>

#include "filesystem.h"

namespace fs = std::filesystem;

#include <vector>

namespace reclib {
namespace utils {
void CreateDirectories(const fs::path& path);
void CreateDirectories(const std::vector<fs::path>& paths);
fs::path SubtractPaths(const fs::path& upper, const fs::path& lower);
}  // namespace utils
}  // namespace reclib

#endif
