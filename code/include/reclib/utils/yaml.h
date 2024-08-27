#ifndef YAML_H
#define YAML_H

#include <yaml-cpp/yaml.h>

#include "reclib/data_types.h"

namespace YAML {
template <typename T>
struct convert<Eigen::Vector<T, 3>> {
  static Node encode(const Eigen::Vector<T, 3>& rhs) {
    Node node;
    node.push_back(rhs.x());
    node.push_back(rhs.y());
    node.push_back(rhs.z());
    return node;
  }

  static bool decode(const Node& node, Eigen::Vector<T, 3>& rhs) {
    if (!node.IsSequence() || node.size() != 3) {
      return false;
    }

    rhs.x() = node[0].as<T>();
    rhs.y() = node[1].as<T>();
    rhs.z() = node[2].as<T>();
    return true;
  }
};
}  // namespace YAML

#endif