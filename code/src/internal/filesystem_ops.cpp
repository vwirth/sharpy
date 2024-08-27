#include <reclib/internal/filesystem_ops.h>

#include <iostream>

void reclib::utils::CreateDirectories(const fs::path& path) {
  if (fs::is_directory(path)) {
    return;
  }

  if (path.has_extension()) {
    CreateDirectories(path.parent_path());
  } else if (!(fs::is_directory(path)) &&
             fs::is_directory(path.parent_path())) {
    try {
      fs::create_directory(path);
    } catch (std::filesystem::filesystem_error const& e) {
      std::cout << "CreateDirectories with path: " << path.string() << ": "
                << e.what() << std::endl;
    } catch (std::bad_alloc const& e) {
      std::cout << "CreateDirectories with path:" << path.string() << ": "
                << e.what() << std::endl;
    }
  } else if (!fs::is_directory(path) && path.string().length() > 0) {
    CreateDirectories(path.parent_path());
    fs::create_directory(path);
  } else {
    std::cout << "invalid path. abort" << std::endl;
  }
}

void reclib::utils::CreateDirectories(const std::vector<fs::path>& paths) {
  for (unsigned int i = 0; i < paths.size(); i++) {
    CreateDirectories(paths[i]);
  }
}

fs::path reclib::utils::SubtractPaths(const fs::path& upper,
                                      const fs::path& lower) {
  fs::path diff_path;
  fs::path base = lower;

  while (base != upper) {
    diff_path = base.stem() / diff_path;
    base = base.parent_path();
  }

  return diff_path;
}
