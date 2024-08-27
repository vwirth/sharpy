#include <reclib/application.h>
#include <reclib/datasets/loader.h>
#include <reclib/opengl/gui.h>
#include <reclib/opengl/quad.h>
#include <reclib/tracking/sharpy_tracker.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <string>

const uint32_t RESOLUTION_H = 1080;
const uint32_t RESOLUTION_W = 1960;
static bool DEBUG = false;
const int NUM_PCA = 23;

// ---------------------------------------
// keyboard variables
// ---------------------------------------
bool request_next_frame = true;

bool process_frame = false;
bool register_frame = false;
bool optimize_stage_0 = false;
bool optimize_stage_1 = false;
bool optimize_stage_2 = false;

bool visualize_wireframe = true;
bool wireframe_mode_changed = true;
bool show_gt_mode_changed = true;
bool show_gt = false;
bool do_benchmark = false;
bool reset = false;
bool visualize_uncertainty = false;
bool vis_blit = false;
bool reset_camera = false;

bool show_kpt = true;
bool show_pred = true;
bool do_screenshot = false;
bool fix_camera = false;

// ---------------------------------------
// state variables
// ---------------------------------------

static reclib::models::Model<reclib::models::MANOConfig> mano_model_right(
    reclib::models::HandType::right);
static reclib::models::Model<reclib::models::MANOConfig> mano_model_left(
    reclib::models::HandType::left);
static reclib::models::Model<reclib::models::MANOConfigAnglePCAGeneric<NUM_PCA>>
    mano_model_pca_right(reclib::models::HandType::right);
static reclib::models::Model<reclib::models::MANOConfigAnglePCAGeneric<NUM_PCA>>
    mano_model_pca_left(reclib::models::HandType::left);

static CpuMat rgb_float;
static reclib::opengl::Timer t;
static int iter = 0;

static fs::path config_path =
    fs::path(RECLIB_LIBRARY_SOURCE_PATH) / fs::path("./configs/sharpy.yaml");
static reclib::Configuration config =
    reclib::Configuration::from_file(config_path);
static std::shared_ptr<reclib::datasets::HandDatasetLoader> loader;
// static reclib::tracking::HandTracker tracker;
static std::shared_ptr<reclib::tracking::HandTracker> tracker;
static reclib::benchmark::SequentialMANOBenchmark benchmark(
    config.subconfig({"Benchmark"}));
static std::string output_base =
    config["Pipeline"]["data_output_folder"].as<std::string>();

// ---------------------------------------
// Functions
// ---------------------------------------

void blit(const reclib::opengl::Texture2D& tex) {
  if (!reclib::opengl::Shader::valid("blit")) {
    reclib::opengl::Shader blitShader("blit", "quad.vs", "blit.fs");
  }
  reclib::opengl::Shader blitShader = reclib::opengl::Shader::find("blit");
  blitShader->bind();
  blitShader->uniform("tex", tex, 0);
  reclib::opengl::Quad::draw();
  blitShader->unbind();
}

void keyboard_callback(int key, int scancode, int action, int mods) {
  if (ImGui::GetIO().WantCaptureKeyboard) return;
  if (mods == GLFW_MOD_SHIFT && key == GLFW_KEY_R && action == GLFW_PRESS)
    reclib::opengl::reload_modified_shaders();
  if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
    reclib::opengl::Context::screenshot("screenshot.png");
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
  }
  if (key == GLFW_KEY_N && action == GLFW_PRESS && !request_next_frame &&
      DEBUG) {
    std::cout << "Request next frame..." << std::endl;
    request_next_frame = true;
  }
  if (key == GLFW_KEY_N && action == GLFW_PRESS && !DEBUG) {
    request_next_frame = !request_next_frame;
  }
  if (key == GLFW_KEY_5 && action == GLFW_PRESS) {
    reset_camera = true;
  }

  if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
    std::cout << "Registering..." << std::endl;
    register_frame = true;
  }
  if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
    std::cout << "Optimizing..." << std::endl;
    optimize_stage_0 = true;
  }
  if (key == GLFW_KEY_3 && action == GLFW_PRESS) {
    std::cout << "Optimizing..." << std::endl;
    optimize_stage_1 = true;
  }
  if (key == GLFW_KEY_4 && action == GLFW_PRESS) {
    std::cout << "Optimizing..." << std::endl;
    optimize_stage_2 = true;
  }
  if (key == GLFW_KEY_6 && action == GLFW_PRESS) {
    show_pred = !show_pred;
  }
  if (key == GLFW_KEY_F2 && action == GLFW_PRESS) {
    do_screenshot = true;
  } else {
    do_screenshot = false;
  }
  if (key == GLFW_KEY_F3 && action == GLFW_PRESS) {
    vis_blit = !vis_blit;
  }
  if (key == GLFW_KEY_F4 && action == GLFW_PRESS) {
    fix_camera = !fix_camera;

    if (fix_camera) {
      reclib::opengl::Camera c = reclib::opengl::current_camera();
      c->pos.x() = 0.35;
      c->pos.y() = -0.33;
    }
  }
  if (key == GLFW_KEY_F5 && action == GLFW_PRESS) {
    std::cout << "Processing..." << std::endl;
    process_frame = true;
  }

  if (key == GLFW_KEY_7 && action == GLFW_PRESS) {
    show_kpt = !show_kpt;
  }
  if (key == GLFW_KEY_8 && action == GLFW_PRESS) {
    visualize_uncertainty = true;
  }
  if (key == GLFW_KEY_9 && action == GLFW_PRESS) {
    visualize_wireframe = !visualize_wireframe;
    wireframe_mode_changed = true;
  }
  if (key == GLFW_KEY_G && action == GLFW_PRESS) {
    show_gt = !show_gt;
    show_gt_mode_changed = true;
  }

  if (key == GLFW_KEY_B && action == GLFW_PRESS) {
    do_benchmark = true;
  }
  if (key == GLFW_KEY_0 && action == GLFW_PRESS) {
    reset = true;
  }
}

void initialize_opengl() {
  // init GL

  reclib::opengl::Context::set_keyboard_callback(keyboard_callback);

  reclib::opengl::ShaderImpl::add_shader_path(RECLIB_SHADER_DIRECTORY);
  reclib::opengl::Shader("phong", "MVP_norm_tc.vs", "phong.fs");
  reclib::opengl::Shader("normals", "MVP_norm_tc.vs", "colorNormals.fs");
  reclib::opengl::Shader("color", "MVP_norm_tc.vs", "color4Uniform.fs");
  reclib::opengl::Shader("colorPV", "MVP_norm_color3.vs", "color3PV.fs");

  reclib::opengl::Camera default_cam = reclib::opengl::current_camera();
  default_cam->pos = vec3(0, 0, 0);
  default_cam->dir = vec3(0, 0, 1);
  default_cam->up = vec3(0, -1, 0);
  default_cam->fix_up_vector = false;
  default_cam->far = 5000;
  reclib::IntrinsicParameters intrinsics = loader->get_intrinsics();
  default_cam->from_intrinsics(intrinsics);
  default_cam->update();

  reclib::opengl::CameraImpl::default_camera_movement_speed = 0.0001;
  reclib::opengl::Shader plane_shader("pointlight_shader", "MVP_norm_tc.vs",
                                      "pointlight3_colorTex.fs");
  fs::path texture_path = RECLIB_DATA_DIRECTORY / fs::path("textures") /
                          fs::path("checkerboard.png");
  reclib::opengl::Texture2D t("checkerboard", texture_path);
  reclib::opengl::Material m("checkerboard");
  m->vec3_map["pl1_pos"] = vec3(0, 0, -1);
  m->float_map["pl1_attenuation"] = 1;

  std::vector<vec3> plane_vertices;
  std::vector<uint32_t> plane_indices;
  std::vector<vec2> plane_tcs;

  plane_vertices.push_back(vec3(1, -1, 0.8));
  plane_vertices.push_back(vec3(-1, 1, 0.8));
  plane_vertices.push_back(vec3(1, 1, 0.8));
  plane_vertices.push_back(vec3(-1, -1, 0.8));

  plane_tcs.push_back(vec2(1, 0));
  plane_tcs.push_back(vec2(0, 1));
  plane_tcs.push_back(vec2(1, 1));
  plane_tcs.push_back(vec2(0, 0));

  plane_indices.push_back(0);
  plane_indices.push_back(3);
  plane_indices.push_back(1);

  plane_indices.push_back(1);
  plane_indices.push_back(2);
  plane_indices.push_back(0);

  reclib::opengl::Geometry g("plane", plane_vertices, plane_indices,
                             std::vector<vec3>(), plane_tcs);
  g->auto_generate_normals();
  reclib::opengl::Mesh mesh("plane", g, m);
  g->register_mesh(mesh);
  reclib::opengl::Drawelement d("plane", plane_shader, mesh);

  glClearColor(1, 1, 1, 1);
  glPointSize(10.f);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);
}

void initialize_globals() {
  bool benchmark_only = config["Benchmark"]["enable"].as<bool>();
  if (benchmark_only) {
    for (auto it : config["Debug"]) {
      it.second = false;
    }
    config["Dataset"]["num_frames"] = -1;
  }

  reclib::Configuration dataset_config = config.subconfig({"Dataset"});
  if (dataset_config["name"].as<std::string>().compare("real") == 0) {
#if HAS_K4A
    loader = std::make_shared<reclib::datasets::AzureLoader>(dataset_config);
#endif
  } else if (dataset_config["name"].as<std::string>().compare("h2o") == 0) {
    loader = std::make_shared<reclib::datasets::H2OLoader>(dataset_config);
  } else if (dataset_config["name"].as<std::string>().compare("h2o-3d") == 0) {
    loader =
        std::make_shared<reclib::datasets::H2O3DLoader>(dataset_config, false);
  } else if (dataset_config["name"].as<std::string>().compare("ho-3d") == 0) {
    loader =
        std::make_shared<reclib::datasets::H2O3DLoader>(dataset_config, true);
  } else {
    throw std::runtime_error("Unknown dataset " +
                             dataset_config["name"].as<std::string>());
  }

  DEBUG = config["Debug"]["stop_through_sequence"].as<bool>();
  reclib::opengl::CameraImpl::set_screenshot_location(
      config["Pipeline"]["data_output_folder"].as<std::string>());
}

void update_config() {
  std::optional<reclib::datasets::MetaData> d = loader->get_metadata();
  std::string seq = "unknown";
  if (d.has_value()) {
    seq = d->seq_path_.string();
    std::replace(seq.begin(), seq.end(), '/', '_');
    std::replace(seq.begin(), seq.end(), '\\', '_');
  }

  tracker->config()["Pipeline"]["data_output_folder"] =
      (fs::path(output_base) / fs::path(seq)).string();
  config["Pipeline"]["data_output_folder"] =
      (fs::path(output_base) / fs::path(seq)).string();
  reclib::opengl::CameraImpl::set_screenshot_location(
      config["Pipeline"]["data_output_folder"].as<std::string>());

  std::cout << "updated screenshot location: "
            << config["Pipeline"]["data_output_folder"] << std::endl;
  std::cout << "Camera location: "
            << reclib::opengl::CameraImpl::screenshot_location << std::endl;
}

void update_user_input() {
  if (tracker->config()["Pipeline"]["save_screenshot"].as<bool>() &&
      rgb_float.rows > 0) {
    reclib::opengl::Context::resize(rgb_float.cols, rgb_float.rows);
  }

  if (reset_camera) {
    reclib::opengl::Camera default_cam = reclib::opengl::current_camera();
    default_cam->pos = vec3(0, 0, 0);
    default_cam->dir = vec3(0, 0, 1);
    default_cam->up = vec3(0, -1, 0);
    default_cam->fix_up_vector = false;
    default_cam->far = 5000;
    default_cam->update();
    reset_camera = false;
  }

  reclib::opengl::CameraImpl::default_input_handler(
      reclib::opengl::Context::frame_time());

  if (t.look_and_reset() > 2000) reclib::opengl::reload_modified_shaders();
  reclib::opengl::current_camera()->update();
}

void update_tracker() {
  if (reset) {
    std::cout << "resetting..." << std::endl;
    tracker->reset();
    benchmark.reset();
    reset = false;
  }
  if (wireframe_mode_changed && config["Debug"]["show_hands"].as<bool>()) {
    for (unsigned int i = 0; i < tracker->hand_states().size(); i++) {
      tracker->hand_states()[i].instance_->gl_instance->wireframe_mode =
          visualize_wireframe;
    }
    wireframe_mode_changed = false;
  }
  CpuMat rgb_map;
  CpuMat depth;
  bool tracker_reset = false;
  bool received_frame = false;
  if (request_next_frame) {
    if (config["Benchmark"]["start_frame"].as<int>() > 0) {
      loader->set_index(config["Benchmark"]["start_frame"].as<int>());
      tracker->set_global_fc(loader->index());
    }

    if (loader->get_input(rgb_map, depth, tracker_reset)) {
      rgb_map.convertTo(rgb_float, CV_32FC3);
      cv::cvtColor(rgb_float, rgb_float, cv::COLOR_BGR2RGB);
      rgb_float = rgb_float / 255;
      if (!reclib::opengl::Texture2D::valid("screenspace_img")) {
        reclib::opengl::Texture2D("screenspace_img", rgb_float.cols,
                                  rgb_float.rows, GL_RGB32F, GL_RGB, GL_FLOAT);
      }
      reclib::opengl::Texture2D t =
          reclib::opengl::Texture2D::find("screenspace_img");
      t->load(rgb_float.ptr());
      reclib::opengl::ImageViewer::instance().add_mat("rgb_float", rgb_float);

      if (tracker_reset) {
        tracker->reset();
        update_config();
      }

      std::vector<int> crop =
          config["Dataset"]["crop_bb"].as<std::vector<int>>();
      cv::Rect crop_fractor(-1, -1, -1, -1);
      if (crop[0] >= 0) {
        crop_fractor.x = crop[0];
        crop_fractor.y = crop[1];
        crop_fractor.width = crop[2] - crop[0];
        crop_fractor.height = crop[3] - crop[1];
      }

      // cv::Rect(400, 400, rgb_map.cols - 600, rgb_map.rows - 600)
      tracker->process_image(rgb_map, depth, loader->get_intrinsics(),
                             crop_fractor);
    } else {
      std::cout << "---------------------------------------------" << std::endl;
      exit(0);
    }

    if (DEBUG) request_next_frame = false;
    received_frame = true;
    iter++;
  }

  if ((DEBUG && register_frame) ||
      (!DEBUG &&
       ((received_frame && (tracker_reset || iter == 1)) || register_frame))) {
    tracker->compute_correspondences(
        reclib::tracking::CorrespondenceMode::REGISTRATION);
    std::vector<bool> res = tracker->register_hand();
    tracker->retry_failed();
    std::cout << "Done Registering..." << std::endl;

    if (register_frame) register_frame = false;
  }

  if ((DEBUG && optimize_stage_0) ||
      (!DEBUG && (received_frame || optimize_stage_0))) {
    tracker->compute_correspondences(
        reclib::tracking::CorrespondenceMode::OPTIMIZATION);
    std::vector<bool> res = tracker->optimize_hand(0);
    tracker->retry_failed();
    std::cout << "Done Optimizing..." << std::endl;

    if (config["Visualization"]["enable"].as<bool>())
      tracker->visualize_uncertainty();
    // }
    if (optimize_stage_0) optimize_stage_0 = false;
  }

  if ((DEBUG && optimize_stage_1) ||
      (!DEBUG && (received_frame || optimize_stage_1))) {
    tracker->compute_correspondences(
        reclib::tracking::CorrespondenceMode::OPTIMIZATION);

    std::vector<bool> res = tracker->optimize_hand(1);
    tracker->retry_failed();
    std::cout << "Done Optimizing..." << std::endl;

    if (config["Visualization"]["enable"].as<bool>())
      tracker->visualize_uncertainty();

    if (optimize_stage_1) optimize_stage_1 = false;
  }

  if ((DEBUG && optimize_stage_2) ||
      (!DEBUG && (received_frame || optimize_stage_2))) {
    tracker->compute_correspondences(
        reclib::tracking::CorrespondenceMode::OPTIMIZATION);

    std::vector<bool> res = tracker->optimize_hand(2);
    std::cout << "Done Optimizing..." << std::endl;
    // if (!res.first || !res.second) {
    //   std::cout << "Stage 1 Optimization failed" << std::endl;
    // }
    if (config["Visualization"]["enable"].as<bool>())
      tracker->visualize_uncertainty();

    if (optimize_stage_2) optimize_stage_2 = false;
  }

  if ((DEBUG && process_frame) ||
      (!DEBUG &&
       ((received_frame && (tracker_reset || iter == 1)) || register_frame))) {
    tracker->compute_all_correspondences();
    std::cout << "Done Processing..." << std::endl;

    if (config["Visualization"]["enable"].as<bool>())
      tracker->visualize_uncertainty();
    if (process_frame) process_frame = false;
  }

  if (visualize_uncertainty) {
    tracker->visualize_uncertainty();
    visualize_uncertainty = false;
  }
}

void update() {
  update_user_input();
  update_tracker();
}

void render() {
  std::vector<std::string> already_rendered;
  for (unsigned int i = 0; i < tracker->hand_states().size(); i++) {
    already_rendered.push_back(
        tracker->hand_states()[i].instance_->gl_instance->name);
    already_rendered.push_back(
        tracker->hand_states()[i].instance_->gl_joints->name);
    already_rendered.push_back(
        tracker->hand_states()[i].instance_->gl_joint_lines->name);
  }

  already_rendered.push_back("plane");
  already_rendered.push_back("corr_points_0");
  already_rendered.push_back("corr_lines_0");
  already_rendered.push_back("corr_points_1");
  already_rendered.push_back("corr_lines_1");
  if (config["Debug"]["show_plane"].as<bool>()) {
    {
      reclib::opengl::Drawelement d =
          reclib::opengl::Drawelement::find("plane");
      reclib::opengl::Texture2D t =
          reclib::opengl::Texture2D::find("checkerboard");
      d->bind();
      t->bind(0);
      d->shader->uniform("color_test", vec4(1, 0, 0, 1));
      d->shader->uniform("diffuse", 0);
      d->draw();
      t->unbind();
      d->unbind();
    }
  }

  for (auto it : reclib::opengl::Drawelement::map) {
    if (it.second->is_grouped) continue;

    if (std::find(already_rendered.begin(), already_rendered.end(), it.first) ==
        already_rendered.end()) {
      it.second->bind();
      it.second->draw();
      it.second->unbind();
    }
  }

  if (tracker->config()["Debug"]["show_hands"].as<bool>()) {
    if (show_pred) {
      for (unsigned int i = 0; i < tracker->hand_states().size(); i++) {
        tracker->hand_states()[i].instance_->gl_instance->bind();
        tracker->hand_states()[i].instance_->gl_instance->draw();
        tracker->hand_states()[i].instance_->gl_instance->unbind();
      }
    }
  }

  if (show_kpt && tracker->config()["Debug"]["show_hands"].as<bool>()) {
    for (unsigned int i = 0; i < tracker->hand_states().size(); i++) {
      tracker->hand_states()[i].instance_->gl_joint_lines->bind();
      tracker->hand_states()[i].instance_->gl_joint_lines->draw();
      tracker->hand_states()[i].instance_->gl_joint_lines->unbind();
      tracker->hand_states()[i].instance_->gl_joints->bind();
      tracker->hand_states()[i].instance_->gl_joints->draw();
      tracker->hand_states()[i].instance_->gl_joints->unbind();
    }
  }

  if (rgb_float.cols > 0 && vis_blit) {
    reclib::opengl::Texture2D t =
        reclib::opengl::Texture2D::find("screenspace_img");
    blit(t);
  }

  if (config["Debug"]["highlight_mesh"].as<bool>())
    glClear(GL_DEPTH_BUFFER_BIT);

  if (tracker->config()["Debug"]["show_hands"].as<bool>()) {
    if (show_pred) {
      for (unsigned int i = 0; i < tracker->hand_states().size(); i++) {
        tracker->hand_states()[i].instance_->gl_instance->bind();
        tracker->hand_states()[i].instance_->gl_instance->draw();
        tracker->hand_states()[i].instance_->gl_instance->unbind();
      }
    }
  }

  if (show_kpt && tracker->config()["Debug"]["show_hands"].as<bool>()) {
    for (unsigned int i = 0; i < tracker->hand_states().size(); i++) {
      tracker->hand_states()[i].instance_->gl_joint_lines->bind();
      tracker->hand_states()[i].instance_->gl_joint_lines->draw();
      tracker->hand_states()[i].instance_->gl_joint_lines->unbind();
      tracker->hand_states()[i].instance_->gl_joints->bind();
      tracker->hand_states()[i].instance_->gl_joints->draw();
      tracker->hand_states()[i].instance_->gl_joints->unbind();
    }
  }

  if (reclib::opengl::Drawelement::valid("corr_points_0")) {
    reclib::opengl::Drawelement p =
        reclib::opengl::Drawelement::find("corr_points_0");
    reclib::opengl::Drawelement l =
        reclib::opengl::Drawelement::find("corr_lines_0");

    l->bind();
    l->draw();
    l->unbind();
    p->bind();
    p->draw();
    p->unbind();
  }
  if (reclib::opengl::Drawelement::valid("corr_points_1")) {
    reclib::opengl::Drawelement p =
        reclib::opengl::Drawelement::find("corr_points_1");
    reclib::opengl::Drawelement l =
        reclib::opengl::Drawelement::find("corr_lines_1");

    l->bind();
    l->draw();
    l->unbind();
    p->bind();
    p->draw();
    p->unbind();
  }
}

int main(int argc, char** argv) {
  t.begin();

  reclib::opengl::ContextParameters params;
  params.width = RESOLUTION_W;
  params.height = RESOLUTION_H;
  params.title = "Mano2RGBD";
  params.floating = GLFW_TRUE;
  params.resizable = GLFW_FALSE;
  params.swap_interval = 1;

  reclib::OpenGLApplication& app = reclib::OpenGLApplication::instance();
  app.set_initialize([]() {
    initialize_globals();
    initialize_opengl();
    tracker = std::make_shared<reclib::tracking::HandTracker>(config);
    update_config();
  });
  app.set_update(update);
  app.set_render(render);
  app.init(params);

  reclib::opengl::ConfigViewer::instance().add_bool("Reset camera",
                                                    &reset_camera);
  reclib::opengl::ConfigViewer::instance().add_config("Config", &config);

  app.start();

  // reset tracker here since the destructor of static variables
  // does not have a specific order
  // and we need to release the tracker first
  tracker.reset();
}
