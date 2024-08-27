#include <reclib/benchmark/mano.h>

float reclib::benchmark::area_under_curve(std::vector<float> y, float min_x,
                                          float max_x) {
  float integral = 0;

  float step = (max_x - min_x) / (float)y.size();
  for (unsigned int i = 0; i < y.size() - 1; i++) {
    float height = y[i];
    float width = step;
    float area = width * height;
    integral += area;
  }

  float normalization = (max_x - min_x);
  return integral / normalization;
}

// -------------------------------------------------------------------------------
// GenericSingleHandMANOBenchmark
// -------------------------------------------------------------------------------

reclib::benchmark::GenericSingleHandMANOBenchmark::
    GenericSingleHandMANOBenchmark()
    : is_left_hand_(false), store_results_(false), num_samples_(0){};
reclib::benchmark::GenericSingleHandMANOBenchmark::
    ~GenericSingleHandMANOBenchmark(){};

void reclib::benchmark::GenericSingleHandMANOBenchmark::reset() {
  num_samples_ = 0;
}

void reclib::benchmark::GenericSingleHandMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred) {}

void reclib::benchmark::GenericSingleHandMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred) {}

void reclib::benchmark::GenericSingleHandMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred) {}

void reclib::benchmark::GenericSingleHandMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred) {}

std::map<std::string, std::map<std::string, float>>
reclib::benchmark::GenericSingleHandMANOBenchmark::get_metrics() const {
  std::map<std::string, std::map<std::string, float>> results;
  return results;
}

void reclib::benchmark::GenericSingleHandMANOBenchmark::print_metrics() const {
  std::map<std::string, std::map<std::string, float>> results = get_metrics();

  std::ios_base::fmtflags f(std::cout.flags());
  const unsigned int line_cap = 100;
  for (auto iter : results) {
    std::string table_header = iter.first;
    std::map<std::string, float> row = iter.second;

    unsigned int max_col_length = 0;
    unsigned int num_cols = 0;
    for (auto col_iter : row) {
      if (col_iter.first.length() > max_col_length) {
        max_col_length = col_iter.first.length();
      }
      std::string val_as_string = std::to_string(col_iter.second);
      if (val_as_string.length() > max_col_length) {
        max_col_length = val_as_string.length();
      }
      num_cols++;
    }

    unsigned int col_width = max_col_length + 1;
    if (num_cols == 1) {
      for (auto col_iter : row) {
        std::cout << table_header << " " << col_iter.first << ":"
                  << std::setw(col_width) << col_iter.second << std::endl;
      }
      continue;
    }
    unsigned int string_width = col_width * num_cols;
    unsigned int string_center = line_cap / 2;
    unsigned int header_half = iter.first.length() / 2;
    std::stringstream header;
    for (unsigned int i = 0; i < string_center - header_half; i++) {
      header << " ";
    }
    header << iter.first;
    for (unsigned int i = 0; i < string_center - header_half; i++) {
      header << " ";
    }

    std::stringstream separator_start;
    for (unsigned int i = 0; i < line_cap; i++) {
      separator_start << "+";
    }

    std::stringstream separator;
    for (unsigned int i = 0; i < line_cap; i++) {
      separator << "-";
    }

    std::cout << separator_start.str() << std::endl;
    std::cout << header.str() << std::endl;
    std::cout << separator.str() << std::endl;
    unsigned int line_size = 0;

    std::stringstream entry;
    for (auto col_iter : row) {
      std::cout << std::setw(col_width) << col_iter.first;
      entry << std::fixed << std::setprecision(3) << std::setw(col_width)
            << col_iter.second;

      line_size += col_width + col_iter.first.length();
      if (line_size >= line_cap) {
        line_size = 0;
        std::cout << std::endl;
        std::cout << entry.str() << std::endl;
        entry.str(std::string());
        std::cout << separator.str() << std::endl;
      }
    }
    if (line_size > 0) {
      line_size = 0;
      std::cout << std::endl;
      std::cout << entry.str() << std::endl;
      entry.str(std::string());
      std::cout << separator.str() << std::endl;
    }
  }

  std::cout.flags(f);
}

// -------------------------------------------------------------------------------
// GenericDoubleHandMANOBenchmark
// -------------------------------------------------------------------------------

reclib::benchmark::GenericDoubleHandMANOBenchmark::
    GenericDoubleHandMANOBenchmark()
    : store_results_(false), num_samples_(0){};
reclib::benchmark::GenericDoubleHandMANOBenchmark::
    ~GenericDoubleHandMANOBenchmark(){};

void reclib::benchmark::GenericDoubleHandMANOBenchmark::reset() {
  num_samples_ = 0;
}

void reclib::benchmark::GenericDoubleHandMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>&
        pred_right) {}

void reclib::benchmark::GenericDoubleHandMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_right) {}

void reclib::benchmark::GenericDoubleHandMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>&
        pred_right) {}

void reclib::benchmark::GenericDoubleHandMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_right) {}

std::map<std::string, std::map<std::string, float>>
reclib::benchmark::GenericDoubleHandMANOBenchmark::get_metrics() const {
  std::map<std::string, std::map<std::string, float>> results;
  return results;
}

void reclib::benchmark::GenericDoubleHandMANOBenchmark::print_metrics() const {
  std::map<std::string, std::map<std::string, float>> results = get_metrics();

  std::ios_base::fmtflags f(std::cout.flags());
  const unsigned int line_cap = 100;
  for (auto iter : results) {
    std::string table_header = iter.first;
    std::map<std::string, float> row = iter.second;

    unsigned int max_col_length = 0;
    unsigned int num_cols = 0;
    for (auto col_iter : row) {
      if (col_iter.first.length() > max_col_length) {
        max_col_length = col_iter.first.length();
      }
      std::string val_as_string = std::to_string(col_iter.second);
      if (val_as_string.length() > max_col_length) {
        max_col_length = val_as_string.length();
      }
      num_cols++;
    }

    unsigned int col_width = max_col_length + 1;
    if (num_cols == 1) {
      for (auto col_iter : row) {
        std::cout << table_header << " " << col_iter.first << ":"
                  << std::setw(col_width) << col_iter.second << std::endl;
      }
      continue;
    }
    unsigned int string_width = col_width * num_cols;
    unsigned int string_center = line_cap / 2;
    unsigned int header_half = iter.first.length() / 2;
    std::stringstream header;
    for (unsigned int i = 0; i < string_center - header_half; i++) {
      header << " ";
    }
    header << iter.first;
    for (unsigned int i = 0; i < string_center - header_half; i++) {
      header << " ";
    }

    std::stringstream separator_start;
    for (unsigned int i = 0; i < line_cap; i++) {
      separator_start << "+";
    }

    std::stringstream separator;
    for (unsigned int i = 0; i < line_cap; i++) {
      separator << "-";
    }

    std::cout << separator_start.str() << std::endl;
    std::cout << header.str() << std::endl;
    std::cout << separator.str() << std::endl;
    unsigned int line_size = 0;

    std::stringstream entry;
    for (auto col_iter : row) {
      std::cout << std::setw(col_width) << col_iter.first;
      entry << std::fixed << std::setprecision(3) << std::setw(col_width)
            << col_iter.second;

      line_size += col_width + col_iter.first.length();
      if (line_size >= line_cap) {
        line_size = 0;
        std::cout << std::endl;
        std::cout << entry.str() << std::endl;
        entry.str(std::string());
        std::cout << separator.str() << std::endl;
      }
    }
    if (line_size > 0) {
      line_size = 0;
      std::cout << std::endl;
      std::cout << entry.str() << std::endl;
      entry.str(std::string());
      std::cout << separator.str() << std::endl;
    }
  }

  std::cout.flags(f);
}

// -------------------------------------------------------------------------------
// SequentialMANOBenchmark
// -------------------------------------------------------------------------------

reclib::benchmark::SequentialMANOBenchmark::SequentialMANOBenchmark(
    const reclib::Configuration& config)
    : config_(config) {
  if (config.exists("H2OBenchmark") && config.b("H2OBenchmark")) {
    // left hand single benchmark
    benchmark_list_single_.push_back(std::make_shared<H2OBenchmark>());
    benchmark_list_single_[benchmark_list_single_.size() - 1]->store_results_ =
        config.b("save");
    // right hand single benchmark
    benchmark_list_single_.push_back(std::make_shared<H2OBenchmark>());
    benchmark_list_single_[benchmark_list_single_.size() - 1]->store_results_ =
        config.b("save");
  }
  if (config.exists("HO3DBenchmark") && config.b("HO3DBenchmark")) {
    // right hand single benchmark
    benchmark_list_single_.push_back(std::make_shared<H2OBenchmark>());
    benchmark_list_single_[benchmark_list_single_.size() - 1]->store_results_ =
        config.b("save");
  }
  if (config.exists("H2O3DBenchmark") && config.b("H2O3DBenchmark")) {
    benchmark_list_double_.push_back(std::make_shared<H2O3DBenchmark>());
    benchmark_list_double_[benchmark_list_double_.size() - 1]->store_results_ =
        config.b("save");
  }
};
reclib::benchmark::SequentialMANOBenchmark::~SequentialMANOBenchmark(){};

void reclib::benchmark::SequentialMANOBenchmark::reset() {
  for (unsigned int i = 0; i < benchmark_list_single_.size(); i++) {
    benchmark_list_single_[i]->reset();
  }
  for (unsigned int i = 0; i < benchmark_list_double_.size(); i++) {
    benchmark_list_double_[i]->reset();
  }
}

template <typename T, typename S>
void reclib::benchmark::SequentialMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<T>& gt,
    const reclib::models::ModelInstance<S>& pred) {
  for (unsigned int i = 0; i < benchmark_list_single_.size(); i++) {
    benchmark_list_single_[i]->evaluate(gt, pred);
  }
}

template <typename T, typename S>
void reclib::benchmark::SequentialMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<T>& gt_left,
    const reclib::models::ModelInstance<T>& gt_right,
    const reclib::models::ModelInstance<S>& pred_left,
    const reclib::models::ModelInstance<S>& pred_right) {
  _RECLIB_ASSERT(gt_left.model.hand_type == reclib::models::HandType::left);
  _RECLIB_ASSERT(gt_right.model.hand_type == reclib::models::HandType::right);
  _RECLIB_ASSERT(pred_left.model.hand_type == reclib::models::HandType::left);
  _RECLIB_ASSERT(pred_right.model.hand_type == reclib::models::HandType::right);

  for (unsigned int i = 0; i < benchmark_list_single_.size(); i++) {
    if (i % 2 == 0) {
      benchmark_list_single_[i]->evaluate(gt_left, pred_left);
    } else if (i % 2 == 1) {
      benchmark_list_single_[i]->evaluate(gt_right, pred_right);
    }
  }

  for (unsigned int i = 0; i < benchmark_list_double_.size(); i++) {
    benchmark_list_double_[i]->evaluate(gt_left, gt_right, pred_left,
                                        pred_right);
  }
}

std::map<std::string, std::map<std::string, float>>
reclib::benchmark::SequentialMANOBenchmark::get_metrics() const {
  std::map<std::string, std::map<std::string, float>> results;

  for (unsigned int i = 0; i < benchmark_list_single_.size(); i++) {
    std::map<std::string, std::map<std::string, float>> tmp =
        benchmark_list_single_[i]->get_metrics();

    for (auto iter : tmp) {
      std::string key = iter.first;
      if (i % 2 == 0) {
        key = key.append("(LEFT)");
      } else {
        key = key.append("(RIGHT)");
      }
      results[key] = iter.second;
    }
  }

  for (unsigned int i = 0; i < benchmark_list_double_.size(); i++) {
    std::map<std::string, std::map<std::string, float>> tmp =
        benchmark_list_double_[i]->get_metrics();

    for (auto iter : tmp) {
      results[iter.first] = iter.second;
    }
  }

  return results;
}

void reclib::benchmark::SequentialMANOBenchmark::print_metrics() const {
  if (benchmark_list_single_.size() > 0) {
    std::cout << "Results from " << benchmark_list_single_[0]->num_samples_
              << " evaluations: " << std::endl;

  } else if (benchmark_list_double_.size() > 0) {
    std::cout << "Results from " << benchmark_list_double_[0]->num_samples_
              << " evaluations: " << std::endl;
  } else {
    std::cout << "Results from 0 "
              << " evaluations: " << std::endl;
  }

  for (unsigned int i = 0; i < benchmark_list_single_.size(); i++) {
    benchmark_list_single_[i]->print_metrics();
  }
  for (unsigned int i = 0; i < benchmark_list_double_.size(); i++) {
    benchmark_list_double_[i]->print_metrics();
  }
}

// ----------------------
// template instantiation
// ----------------------
template void reclib::benchmark::SequentialMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred);
template void reclib::benchmark::SequentialMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred);
template void reclib::benchmark::SequentialMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred);
template void reclib::benchmark::SequentialMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred);

template void reclib::benchmark::SequentialMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>&
        pred_right);
template void reclib::benchmark::SequentialMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_right);
template void reclib::benchmark::SequentialMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>&
        pred_right);
template void reclib::benchmark::SequentialMANOBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_right);
// -------------------------------------------------------------------------------
// H2OBenchmark
// -------------------------------------------------------------------------------

const std::vector<float> reclib::benchmark::H2OBenchmark::PCK_THRESHOLDS = {
    0, 10, 20, 30, 40, 50, 60, 70, 80};  // in millimeters

const std::vector<float>
    reclib::benchmark::H2OBenchmark::PCK_SINGLE_THRESHOLDS = {
        15};  // in millimeters

reclib::benchmark::H2OBenchmark::H2OBenchmark()
    : accumulated_pck_3d_(PCK_THRESHOLDS.size()),
      accumulated_pck_single_3d_(PCK_SINGLE_THRESHOLDS.size()),
      num_evaluations_(0){};
reclib::benchmark::H2OBenchmark::~H2OBenchmark() {
  std::fill(accumulated_pck_3d_.begin(), accumulated_pck_3d_.end(), 0);
  std::fill(accumulated_pck_single_3d_.begin(),
            accumulated_pck_single_3d_.end(), 0);
};

void reclib::benchmark::H2OBenchmark::reset() {
  std::fill(accumulated_pck_3d_.begin(), accumulated_pck_3d_.end(), 0);
  std::fill(accumulated_pck_single_3d_.begin(),
            accumulated_pck_single_3d_.end(), 0);
  std::fill(accumulated_euclidean_joint_dist_.begin(),
            accumulated_euclidean_joint_dist_.end(), 0);
  num_evaluations_ = 0;
  num_samples_ = 0;
}

template <typename T, typename S>
void reclib::benchmark::H2OBenchmark::t_evaluate(
    const reclib::models::ModelInstance<T>& gt,
    const reclib::models::ModelInstance<S>& pred) {
  if (gt.model.hand_type == reclib::models::HandType::left) {
    is_left_hand_ = true;
  } else {
    is_left_hand_ = false;
  }
  num_samples_++;

  // if (is_left_hand_) {
  //   std::cout << "gt: " << gt.pose() << std::endl;
  //   std::cout << "pred: " << pred.pose() << std::endl;
  // } else {
  //   std::cout << "right hand benchmark..." << std::endl;
  // }

  // compute the 3d PCK
  // extracted from: https://arxiv.org/pdf/1904.05349.pdf
  unsigned int all_joints = gt.joints().rows();

  // positive joints counter, sorted by pck threshold
  std::vector<unsigned int> positive_joints(PCK_THRESHOLDS.size());
  std::fill(positive_joints.begin(), positive_joints.end(), 0);

  std::vector<unsigned int> positive_joints_single(
      PCK_SINGLE_THRESHOLDS.size());
  std::fill(positive_joints_single.begin(), positive_joints_single.end(), 0);

  if (accumulated_euclidean_joint_dist_.size() <
      (unsigned int)gt.joints().rows()) {
    accumulated_euclidean_joint_dist_.resize((unsigned int)gt.joints().rows());
    std::fill(accumulated_euclidean_joint_dist_.begin(),
              accumulated_euclidean_joint_dist_.end(), 0);
  }

  for (unsigned int i = 0; i < gt.joints().rows(); i++) {
    vec3 j_gt = gt.joints().row(i).transpose();
    vec3 j_pred = pred.joints().row(i).transpose();

    if (0) {
      std::string tag = "";
      if (is_left_hand_) {
        tag = "left";
      } else {
        tag = "right";
      }
      if (!reclib::opengl::Drawelement::valid("j_gt_" + tag + "_" +
                                              std::to_string(i))) {
        reclib::opengl::Material mat("j_gt_" + tag + "_" + std::to_string(i));
        mat->vec4_map["color"] = vec4(1, 0, 0, 1);
        reclib::opengl::Shader s = reclib::opengl::Shader::find("color");

        reclib::opengl::Drawelement d =
            reclib::opengl::DrawelementImpl::from_geometry(
                "j_gt_" + tag + "_" + std::to_string(i), s, mat, false,
                std::vector<vec3>({j_gt}));
        d->mesh->primitive_type = GL_POINTS;
        d->add_pre_draw_func("pointsize", [&]() { glPointSize(10.f); });
        d->add_post_draw_func("pointsize", [&]() { glPointSize(1.f); });
      }
      if (!reclib::opengl::Drawelement::valid("j_pred_" + tag + "_" +
                                              std::to_string(i))) {
        reclib::opengl::Material mat("j_pred_" + tag + "_" + std::to_string(i));
        mat->vec4_map["color"] = vec4(0, 1, 1, 1);
        reclib::opengl::Shader s = reclib::opengl::Shader::find("color");

        reclib::opengl::Drawelement d =
            reclib::opengl::DrawelementImpl::from_geometry(
                "j_pred_" + tag + "_" + std::to_string(i), s, mat, false,
                std::vector<vec3>({j_pred}));
        d->mesh->primitive_type = GL_POINTS;
        d->add_pre_draw_func("pointsize", [&]() { glPointSize(10.f); });
        d->add_post_draw_func("pointsize", [&]() { glPointSize(1.f); });
      }
      if (!reclib::opengl::Drawelement::valid("j_gt_pred_" + tag + "_" +
                                              std::to_string(i))) {
        reclib::opengl::Material mat("j_gt_pred_" + tag + "_" +
                                     std::to_string(i));
        mat->vec4_map["color"] = vec4(0, 0, 0, 1);
        reclib::opengl::Shader s = reclib::opengl::Shader::find("color");

        reclib::opengl::Drawelement d =
            reclib::opengl::DrawelementImpl::from_geometry(
                "j_gt_pred_" + tag + "_" + std::to_string(i), s, mat, false,
                std::vector<vec3>({j_gt, j_pred}));
        d->mesh->primitive_type = GL_LINES;
        d->add_pre_draw_func("linewidth", [&]() { glLineWidth(2.f); });
        d->add_post_draw_func("linewidth", [&]() { glLineWidth(1.f); });
      }
    }

    vec3 dist = j_gt - j_pred;
    accumulated_euclidean_joint_dist_[i] += dist.norm() * 1000;
    // apparently we compute the MEAN distance, not the norm... whatever
    float mean_dist = (1.f / 3.f) * dist.cwiseAbs().sum();

    mean_dist *= 1000;  // meter -> millimeter

    int positive_index = -1;
    for (unsigned int j = 0; j < PCK_THRESHOLDS.size(); j++) {
      if (mean_dist < PCK_THRESHOLDS[j]) {
        // since the pck thresholds are sorted
        // we can stop as soon as we fall below the current threshold in the
        // loop and break here
        positive_index = j;
        break;
      }
    }
    // increment all positive joints >= the first positive index
    if (positive_index >= 0) {
      for (unsigned int j = positive_index; j < PCK_THRESHOLDS.size(); j++) {
        positive_joints[j]++;
      }
    }

    positive_index = -1;
    for (unsigned int j = 0; j < PCK_SINGLE_THRESHOLDS.size(); j++) {
      if (mean_dist < PCK_SINGLE_THRESHOLDS[j]) {
        // since the pck thresholds are sorted
        // we can stop as soon as we fall below the current threshold in the
        // loop and break here
        positive_index = j;
        break;
      }
    }
    // increment all positive joints >= the first positive index
    if (positive_index >= 0) {
      for (unsigned int j = positive_index; j < PCK_SINGLE_THRESHOLDS.size();
           j++) {
        positive_joints_single[j]++;
      }
    }
  }

  for (unsigned int i = 0; i < positive_joints.size(); i++) {
    float accuracy = (float)positive_joints[i] / (float)all_joints;
    accumulated_pck_3d_[i] += accuracy;
  }

  for (unsigned int i = 0; i < positive_joints_single.size(); i++) {
    float accuracy = (float)positive_joints_single[i] / (float)all_joints;
    accumulated_pck_single_3d_[i] += accuracy;
  }

  num_evaluations_++;
}

void reclib::benchmark::H2OBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred) {
  t_evaluate<reclib::models::MANOConfig, reclib::models::MANOConfig>(gt, pred);
}

void reclib::benchmark::H2OBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred) {
  t_evaluate<reclib::models::MANOConfigPCA, reclib::models::MANOConfigPCA>(
      gt, pred);
}

void reclib::benchmark::H2OBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred) {
  t_evaluate<reclib::models::MANOConfigPCA, reclib::models::MANOConfig>(gt,
                                                                        pred);
}

void reclib::benchmark::H2OBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred) {
  t_evaluate<reclib::models::MANOConfig, reclib::models::MANOConfigPCA>(gt,
                                                                        pred);
}

std::map<std::string, std::map<std::string, float>>
reclib::benchmark::H2OBenchmark::get_metrics() const {
  std::map<std::string, std::map<std::string, float>> results_all;
  if (num_evaluations_ == 0) return results_all;

  std::vector<float> AUC_values;
  float total_euclidean_joint_dist = 0;
  std::string handedness;
  if (is_left_hand_) {
    handedness = "(LEFT)";
  } else {
    handedness = "(RIGHT)";
  }

  {
    std::map<std::string, float> results;
    for (unsigned int i = 0; i < PCK_THRESHOLDS.size(); i++) {
      float pck_accum = accumulated_pck_3d_[i];
      float mean_pck = pck_accum / (float)num_evaluations_;

      std::string key = std::to_string(PCK_THRESHOLDS[i]);
      results[key] = mean_pck;
      AUC_values.push_back(mean_pck);
    }
    for (unsigned int i = 0; i < PCK_SINGLE_THRESHOLDS.size(); i++) {
      float pck_accum = accumulated_pck_single_3d_[i];
      float mean_pck = pck_accum / (float)num_evaluations_;

      std::string key = std::to_string(PCK_SINGLE_THRESHOLDS[i]);
      if (results.find(key) == results.end()) {
        results[key] = mean_pck;
      }
    }

    results_all["3DPCK(mm)" + handedness] = results;
  }

  // {
  //   std::map<std::string, float> results;
  //   for (unsigned int i = 0; i < PCK_SINGLE_THRESHOLDS.size(); i++) {
  //     float pck_accum = accumulated_pck_single_3d_[i];
  //     float mean_pck = pck_accum / (float)num_evaluations_;

  //     std::string key = std::to_string(PCK_SINGLE_THRESHOLDS[i]);
  //     results[key] = mean_pck;
  //   }
  //   results_all["3DPCK(mm)" + handedness] = results;
  // }

  {
    std::map<std::string, float> results;
    for (unsigned int i = 0; i < accumulated_euclidean_joint_dist_.size();
         i++) {
      float euclidean_joint_dist = accumulated_euclidean_joint_dist_[i];
      float mean_euclidean_joint_dist =
          euclidean_joint_dist / (float)num_evaluations_;

      std::string key = std::to_string(i);
      results[key] = mean_euclidean_joint_dist;
      total_euclidean_joint_dist += mean_euclidean_joint_dist;
    }

    total_euclidean_joint_dist /=
        (float)accumulated_euclidean_joint_dist_.size();
    results["Total"] = total_euclidean_joint_dist;
    results_all["EuclideanDist(mm)" + handedness] = results;
  }

  float auc = area_under_curve(AUC_values, PCK_THRESHOLDS[0],
                               PCK_THRESHOLDS[PCK_THRESHOLDS.size() - 1]);
  results_all["AUC" + handedness] =
      std::map<std::string, float>({{"Total", auc}});

  return results_all;
}

// ----------------------
// template instantiation
// ----------------------
template void reclib::benchmark::H2OBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred);
template void reclib::benchmark::H2OBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred);
template void reclib::benchmark::H2OBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred);
template void reclib::benchmark::H2OBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred);

// -------------------------------------------------------------------------------
// HO3DBenchmark
// -------------------------------------------------------------------------------

const std::vector<float> reclib::benchmark::HO3DBenchmark::PCK_THRESHOLDS = {
    0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500};  // in millimeters

reclib::benchmark::HO3DBenchmark::HO3DBenchmark(){};
reclib::benchmark::HO3DBenchmark::~HO3DBenchmark(){};

void reclib::benchmark::HO3DBenchmark::reset() { num_samples_ = 0; }

template <typename T, typename S>
void reclib::benchmark::HO3DBenchmark::t_evaluate(
    const reclib::models::ModelInstance<T>& gt,
    const reclib::models::ModelInstance<S>& pred) {
  if (gt.model.hand_type == reclib::models::HandType::left) {
    is_left_hand_ = true;
  } else {
    is_left_hand_ = false;
  }

  // https://github.com/shreyashampali/HOnnotate/blob/master/eval/eval3DKps.py
  // procrustes aligned 3DPCK

  // f-score:
  // https://codalab.lisn.upsaclay.fr/competitions/4393#learn_the_details-evaluation
}

void reclib::benchmark::HO3DBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred) {
  t_evaluate<reclib::models::MANOConfig, reclib::models::MANOConfigPCA>(gt,
                                                                        pred);
}

void reclib::benchmark::HO3DBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred) {
  t_evaluate<reclib::models::MANOConfigPCA, reclib::models::MANOConfig>(gt,
                                                                        pred);
}

void reclib::benchmark::HO3DBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred) {
  t_evaluate<reclib::models::MANOConfig, reclib::models::MANOConfig>(gt, pred);
}

void reclib::benchmark::HO3DBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred) {
  t_evaluate<reclib::models::MANOConfigPCA, reclib::models::MANOConfigPCA>(
      gt, pred);
}

std::map<std::string, std::map<std::string, float>>
reclib::benchmark::HO3DBenchmark::get_metrics() const {
  return std::map<std::string, std::map<std::string, float>>();
}

// ----------------------
// template instantiation
// ----------------------
template void reclib::benchmark::HO3DBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred);
template void reclib::benchmark::HO3DBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred);
template void reclib::benchmark::HO3DBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred);
template void reclib::benchmark::HO3DBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred);

// -------------------------------------------------------------------------------
// H2O3DBenchmark
// -------------------------------------------------------------------------------

reclib::benchmark::H2O3DBenchmark::H2O3DBenchmark()
    : accumulated_mrrpe_(0.f),
      num_evaluations_mpjpe_(0),
      num_evaluations_mrrpe_(0){};
reclib::benchmark::H2O3DBenchmark::~H2O3DBenchmark(){};

void reclib::benchmark::H2O3DBenchmark::reset() {
  std::fill(accumulated_mpjpe_left_.begin(), accumulated_mpjpe_left_.end(), 0);
  std::fill(accumulated_mpjpe_right_.begin(), accumulated_mpjpe_right_.end(),
            0);
  num_evaluations_mpjpe_ = 0;
  num_evaluations_mrrpe_ = 0;
  num_samples_ = 0;
}

template <typename T, typename S>
void reclib::benchmark::H2O3DBenchmark::t_evaluate(
    const reclib::models::ModelInstance<T>& gt_left,
    const reclib::models::ModelInstance<T>& gt_right,
    const reclib::models::ModelInstance<S>& pred_left,
    const reclib::models::ModelInstance<S>& pred_right) {
  _RECLIB_ASSERT(gt_left.model.hand_type == reclib::models::HandType::left);
  _RECLIB_ASSERT(gt_right.model.hand_type == reclib::models::HandType::right);
  _RECLIB_ASSERT(pred_left.model.hand_type == reclib::models::HandType::left);
  _RECLIB_ASSERT(pred_right.model.hand_type == reclib::models::HandType::right);

  if (accumulated_mpjpe_left_.size() == 0) {
    accumulated_mpjpe_left_.resize(gt_left.joints().rows());
    std::fill(accumulated_mpjpe_left_.begin(), accumulated_mpjpe_left_.end(),
              0);
  }

  if (accumulated_mpjpe_right_.size() == 0) {
    accumulated_mpjpe_right_.resize(gt_right.joints().rows());
    std::fill(accumulated_mpjpe_right_.begin(), accumulated_mpjpe_right_.end(),
              0);
  }

  num_samples_++;

  auto mpjpe_func = [&](const reclib::models::ModelInstance<T>& g,
                        const reclib::models::ModelInstance<S>& p) {
    // compute the MPJPE and MRRPE
    // extracted from:
    // https://github.com/facebookresearch/InterHand2.6M/blob/main/data/InterHand2.6M/dataset.py
    unsigned int all_joints = g.joints().rows();

    vec3 j_root_gt = g.joints().row(0).transpose();
    vec3 j_root_pred = p.joints().row(0).transpose();

    for (unsigned int i = 0; i < g.joints().rows(); i++) {
      vec3 j_gt_rootnorm = g.joints().row(i).transpose() - j_root_gt;
      vec3 j_pred_rootnorm = p.joints().row(i).transpose() - j_root_pred;
      vec3 dist = j_gt_rootnorm - j_pred_rootnorm;

      if (g.model.hand_type == reclib::models::HandType::left) {
        accumulated_mpjpe_left_[i] +=
            dist.norm() * 1000;  // meter -> millimeter
      } else {
        accumulated_mpjpe_right_[i] +=
            dist.norm() * 1000;  // meter -> millimeter
      }
    }
  };

  mpjpe_func(gt_left, pred_left);
  mpjpe_func(gt_right, pred_right);

  vec3 j_root_gt_left = gt_left.joints().row(0).transpose();
  vec3 j_root_pred_left = pred_left.joints().row(0).transpose();

  vec3 j_root_gt_right = gt_right.joints().row(0).transpose();
  vec3 j_root_pred_right = pred_right.joints().row(0).transpose();

  accumulated_mrrpe_ += ((j_root_pred_left - j_root_pred_right) -
                         (j_root_gt_left - j_root_gt_right))
                            .norm();
  num_evaluations_mrrpe_++;
  num_evaluations_mpjpe_++;
}

void reclib::benchmark::H2O3DBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>&
        pred_right) {
  t_evaluate<reclib::models::MANOConfig, reclib::models::MANOConfig>(
      gt_left, gt_right, pred_left, pred_right);
}

void reclib::benchmark::H2O3DBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_right) {
  t_evaluate<reclib::models::MANOConfigPCA, reclib::models::MANOConfigPCA>(
      gt_left, gt_right, pred_left, pred_right);
}

void reclib::benchmark::H2O3DBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>&
        pred_right) {
  t_evaluate<reclib::models::MANOConfigPCA, reclib::models::MANOConfig>(
      gt_left, gt_right, pred_left, pred_right);
}

void reclib::benchmark::H2O3DBenchmark::evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_right) {
  t_evaluate<reclib::models::MANOConfig, reclib::models::MANOConfigPCA>(
      gt_left, gt_right, pred_left, pred_right);
}

std::map<std::string, std::map<std::string, float>>
reclib::benchmark::H2O3DBenchmark::get_metrics() const {
  std::map<std::string, std::map<std::string, float>> results_all;

  if (num_evaluations_mpjpe_ == 0) return results_all;

  {
    std::map<std::string, float> results;
    float mpjpe_total_left = 0;

    for (unsigned int i = 0; i < accumulated_mpjpe_left_.size(); i++) {
      float mpjpe_left_accum = accumulated_mpjpe_left_[i];

      float mean_left = mpjpe_left_accum / (float)num_evaluations_mpjpe_;
      mpjpe_total_left += mean_left;

      std::string key_left = "J" + std::to_string(i);
      results[key_left] = mean_left;
    }
    mpjpe_total_left /= (float)accumulated_mpjpe_left_.size();
    results["Total"] = mpjpe_total_left;
    results_all["MPJPE(mm)(LEFT)"] = results;
  }

  {
    std::map<std::string, float> results;
    float mpjpe_total_right = 0;
    for (unsigned int i = 0; i < accumulated_mpjpe_left_.size(); i++) {
      float mpjpe_right_accum = accumulated_mpjpe_right_[i];
      float mean_right = mpjpe_right_accum / (float)num_evaluations_mpjpe_;
      mpjpe_total_right += mean_right;

      std::string key_right = "J" + std::to_string(i);
      results[key_right] = mean_right;
    }
    mpjpe_total_right /= (float)accumulated_mpjpe_left_.size();
    results["Total"] = mpjpe_total_right;
    results_all["MPJPE(mm)(RIGHT)"] = results;
  }

  float mean_rrpe = accumulated_mrrpe_ / num_evaluations_mrrpe_;
  results_all["MRRPE(mm)"] =
      std::map<std::string, float>({{"Total", mean_rrpe}});

  return results_all;
}

// ----------------------
// template instantiation
// ----------------------
template void reclib::benchmark::H2O3DBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>&
        pred_right);
template void reclib::benchmark::H2O3DBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_right);
template void reclib::benchmark::H2O3DBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>&
        pred_right);
template void reclib::benchmark::H2O3DBenchmark::t_evaluate(
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_left,
    const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
        pred_right);