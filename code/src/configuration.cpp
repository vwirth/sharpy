#include <reclib/configuration.h>

#include <filesystem>
#include <fstream>

// ------------------------------------------------------------
// ------------------------------------------------------------
// CorrespondencesConfiguration
// ------------------------------------------------------------
// ------------------------------------------------------------

reclib::Configuration reclib::Configuration::from_file(
    const std::filesystem::path& filepath,
    const std::vector<std::string>& prefixes) {
  YAML::Node node = YAML::LoadFile(filepath.string());
  reclib::Configuration config;
  config.yaml_ = node;
  config.prefixes_ = prefixes;
  config.loading_path_ = filepath;
  return config;
}

void reclib::Configuration::reload() {
  if (fs::exists(loading_path_)) yaml_ = YAML::LoadFile(loading_path_.string());
}

void reclib::Configuration::to_file(std::filesystem::path filepath) {
  if (!fs::exists(filepath)) {
    if (fs::exists(loading_path_)) {
      filepath = loading_path_;
    } else {
      std::cout << "[Error] Can not save configuration: Invalid path "
                << filepath << std::endl;
      return;
    }
  }
  std::ofstream fout(filepath.string());
  YAML::Node cur = yaml_;
  for (unsigned int i = 0; i < prefixes_.size(); i++) {
    cur = cur[prefixes_[i]];
  }
  fout << cur;
}
