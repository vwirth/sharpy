#ifndef RECLIB_BENCHMARK_MANO_H
#define RECLIB_BENCHMARK_MANO_H

#include <map>

#include "reclib/configuration.h"
#include "reclib/models/smpl.h"

namespace reclib {

namespace benchmark {

float area_under_curve(std::vector<float> y, float min_x, float max_x);

class GenericSingleHandMANOBenchmark {
 protected:
  bool is_left_hand_;

 public:
  bool store_results_;
  int num_samples_;

  GenericSingleHandMANOBenchmark();
  virtual ~GenericSingleHandMANOBenchmark();

  virtual void reset();

  virtual void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred);

  virtual void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred);

  virtual void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred);

  virtual void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred);

  virtual std::map<std::string, std::map<std::string, float>> get_metrics()
      const;
  virtual void print_metrics() const;
};

class GenericDoubleHandMANOBenchmark {
 public:
  bool store_results_;
  int num_samples_;

  GenericDoubleHandMANOBenchmark();
  virtual ~GenericDoubleHandMANOBenchmark();

  virtual void reset();

  virtual void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>&
          pred_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>&
          pred_right);

  virtual void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          gt_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          gt_right,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          pred_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          pred_right);

  virtual void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          gt_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          gt_right,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>&
          pred_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>&
          pred_right);

  virtual void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          pred_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          pred_right);

  virtual std::map<std::string, std::map<std::string, float>> get_metrics()
      const;
  virtual void print_metrics() const;
};

class SequentialMANOBenchmark {
 private:
  reclib::Configuration config_;

  // we need a vector of shared pointers instead of only the objects themselves
  // see https://en.wikipedia.org/wiki/Object_slicing
  std::vector<std::shared_ptr<GenericSingleHandMANOBenchmark>>
      benchmark_list_single_;
  std::vector<std::shared_ptr<GenericDoubleHandMANOBenchmark>>
      benchmark_list_double_;

 public:
  SequentialMANOBenchmark(const reclib::Configuration& config);
  ~SequentialMANOBenchmark();

  void reset();

  template <typename T, typename S>
  void evaluate(const reclib::models::ModelInstance<T>& gt,
                const reclib::models::ModelInstance<S>& pred);

  template <typename T, typename S>
  void evaluate(const reclib::models::ModelInstance<T>& gt_left,
                const reclib::models::ModelInstance<T>& gt_right,
                const reclib::models::ModelInstance<S>& pred_left,
                const reclib::models::ModelInstance<S>& pred_right);

  std::map<std::string, std::map<std::string, float>> get_metrics() const;
  void print_metrics() const;
};

class H2OBenchmark : public GenericSingleHandMANOBenchmark {
  static const std::vector<float> PCK_THRESHOLDS;
  static const std::vector<float> PCK_SINGLE_THRESHOLDS;

  std::vector<float>
      accumulated_pck_3d_;  // accumulate over all 'evaluate' calls
  std::vector<float>
      accumulated_pck_single_3d_;  // accumulate over all 'evaluate' calls
  std::vector<float> accumulated_euclidean_joint_dist_;
  unsigned int num_evaluations_;

  template <typename T, typename S>
  void t_evaluate(const reclib::models::ModelInstance<T>& gt,
                  const reclib::models::ModelInstance<S>& pred);

 public:
  H2OBenchmark();
  ~H2OBenchmark();

  void reset() override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred)
      override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred)
      override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred)
      override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred)
      override;

  std::map<std::string, std::map<std::string, float>> get_metrics()
      const override;
};

// unfortunately, this benchmark only supports
// hand models with 21 joins vs the MANO model that has only 16 joints
// so we can not use it
class HO3DBenchmark : public GenericSingleHandMANOBenchmark {
  static const std::vector<float> PCK_THRESHOLDS;

  template <typename T, typename S>
  void t_evaluate(const reclib::models::ModelInstance<T>& gt,
                  const reclib::models::ModelInstance<S>& pred);

 public:
  HO3DBenchmark();
  ~HO3DBenchmark();

  void reset() override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred)
      override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred)
      override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& pred)
      override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& gt,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>& pred)
      override;

  std::map<std::string, std::map<std::string, float>> get_metrics()
      const override;
};

class H2O3DBenchmark : public GenericDoubleHandMANOBenchmark {
  std::vector<float> accumulated_mpjpe_left_;
  std::vector<float> accumulated_mpjpe_right_;
  float accumulated_mrrpe_;
  unsigned int num_evaluations_mpjpe_;
  unsigned int num_evaluations_mrrpe_;

  template <typename T, typename S>
  void t_evaluate(const reclib::models::ModelInstance<T>& gt_left,
                  const reclib::models::ModelInstance<T>& gt_right,
                  const reclib::models::ModelInstance<S>& pred_left,
                  const reclib::models::ModelInstance<S>& pred_right);

 public:
  H2O3DBenchmark();
  ~H2O3DBenchmark();

  void reset() override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>&
          pred_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>&
          pred_right) override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          gt_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          gt_right,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          pred_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          pred_right) override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          gt_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          gt_right,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>&
          pred_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>&
          pred_right) override;

  void evaluate(
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfig>& gt_right,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          pred_left,
      const reclib::models::ModelInstance<reclib::models::MANOConfigPCA>&
          pred_right) override;

  std::map<std::string, std::map<std::string, float>> get_metrics()
      const override;
};

}  // namespace benchmark
}  // namespace reclib

#endif