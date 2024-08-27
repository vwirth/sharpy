#ifndef RECLIB_CONFIGURATION_H
#define RECLIB_CONFIGURATION_H

#include <yaml-cpp/yaml.h>

#include <iostream>

#include "reclib/data_types.h"
#include "reclib/internal/filesystem.h"

namespace reclib {
struct Configuration {
 public:
  fs::path loading_path_;
  YAML::Node yaml_;
  std::vector<std::string> prefixes_;

  Configuration(){};

  Configuration(std::string& prefix) : prefixes_({prefix}){};

  Configuration(const std::vector<std::string>& prefixes)
      : prefixes_({prefixes}){};
  Configuration(
      const YAML::Node& yaml,
      const std::vector<std::string>& prefixes = std::vector<std::string>())
      : yaml_(yaml), prefixes_({prefixes}){};

  Configuration(const Configuration& other) {
    yaml_ = YAML::Clone(other.yaml_);
    prefixes_ = other.prefixes_;
  }

  Configuration& operator=(const Configuration& other) {
    this->yaml_ = YAML::Clone(other.yaml_);
    this->prefixes_ = other.prefixes_;
    return *this;
  }

  Configuration clone() {
    Configuration clone;
    clone.prefixes_ = prefixes_;
    clone.yaml_ = YAML::Clone(yaml_);
    return clone;
  }

  Configuration subconfig(const std::vector<std::string>& prefixes,
                          bool keep_original_yaml = false) const {
    std::vector<std::string> merged_prefixes;
    merged_prefixes.insert(merged_prefixes.end(), prefixes_.begin(),
                           prefixes_.end());
    merged_prefixes.insert(merged_prefixes.end(), prefixes.begin(),
                           prefixes.end());

    Configuration sub;
    if (keep_original_yaml) {
      sub.prefixes_ = prefixes;
      sub.yaml_ = YAML::Clone(yaml_);
    } else {
      YAML::Node cur = YAML::Clone(yaml_);
      for (unsigned int i = 0; i < merged_prefixes.size(); i++) {
        cur = cur[merged_prefixes[i]];
      }
      sub.yaml_ = cur;
    }

    return sub;
  }

  auto operator[](const std::string& key) const {
    if (!exists(key)) {
      throw std::runtime_error("Configuration: Key does not exist: " +
                               std::string(key));
    }
    return (yaml_[key]);
  }
  YAML::Node& get() { return yaml_; }

  virtual YAML::Node get(const std::string& key) const {
    if (!exists(key)) {
      throw std::runtime_error("Configuration: Key does not exist: " +
                               std::string(key));
    }
    YAML::Node cur;
    if (prefixes_.size() > 0) {
      cur = YAML::Clone(yaml_);
      for (unsigned int i = 0; i < prefixes_.size(); i++) {
        cur = cur[prefixes_[i]];
      }
    } else {
      cur = yaml_;
    }
    return cur[key];
  }

  virtual bool exists(const std::string& key) const {
    try {
      YAML::Node cur;
      if (prefixes_.size() > 0) {
        cur = YAML::Clone(yaml_);
        for (unsigned int i = 0; i < prefixes_.size(); i++) {
          cur = cur[prefixes_[i]];
        }
      } else {
        cur = yaml_;
      }

      YAML::Node tmp = cur[key];
      return !tmp.IsNull();
    } catch (YAML::InvalidNode& e) {
      return false;
    }

    return true;
  }

  bool b(const char* key) const {
    if (!exists(key)) {
      throw std::runtime_error("Configuration: Key does not exist: " +
                               std::string(key));
    }
    YAML::Node cur;
    if (prefixes_.size() > 0) {
      cur = YAML::Clone(yaml_);
      for (unsigned int i = 0; i < prefixes_.size(); i++) {
        cur = cur[prefixes_[i]];
      }
    } else {
      cur = yaml_;
    }
    return cur[key].as<bool>();
  }

  int i(const char* key) const {
    if (!exists(key)) {
      throw std::runtime_error("Configuration: Key does not exist: " +
                               std::string(key));
    }
    YAML::Node cur;
    if (prefixes_.size() > 0) {
      cur = YAML::Clone(yaml_);
      for (unsigned int i = 0; i < prefixes_.size(); i++) {
        cur = cur[prefixes_[i]];
      }
    } else {
      cur = yaml_;
    }
    return cur[key].as<int>();
  }

  unsigned int ui(const char* key) const {
    if (!exists(key)) {
      throw std::runtime_error("Configuration: Key does not exist: " +
                               std::string(key));
    }
    YAML::Node cur;
    if (prefixes_.size() > 0) {
      cur = YAML::Clone(yaml_);
      for (unsigned int i = 0; i < prefixes_.size(); i++) {
        cur = cur[prefixes_[i]];
      }
    } else {
      cur = yaml_;
    }
    return cur[key].as<unsigned int>();
  }

  float f(const char* key) const {
    if (!exists(key)) {
      throw std::runtime_error("Configuration: Key does not exist: " +
                               std::string(key));
    }
    YAML::Node cur;
    if (prefixes_.size() > 0) {
      cur = YAML::Clone(yaml_);
      for (unsigned int i = 0; i < prefixes_.size(); i++) {
        cur = cur[prefixes_[i]];
      }
    } else {
      cur = yaml_;
    }
    return cur[key].as<float>();
  }

  static reclib::Configuration from_file(
      const std::filesystem::path& filepath,
      const std::vector<std::string>& prefixes = std::vector<std::string>());
  virtual void to_file(std::filesystem::path filepath = "");
  void reload();

  friend std::ostream& operator<<(std::ostream& os, const YAML::Node& n) {
    os << std::endl;
    for (const auto& kv : n) {
      os << "---" << kv.first.as<std::string>() << ":";
      os << kv.second;
    }
    return os;
  }
  friend std::ostream& operator<<(std::ostream& os, const Configuration& q) {
    YAML::Node cur = q.yaml_;
    os << std::endl;
    for (unsigned int i = 0; i < q.prefixes_.size(); i++) {
      cur = cur[q.prefixes_[i]];
    }
    os << cur << std::endl;
    return os;
  }
};  // namespace reclib
}  // namespace reclib

#endif