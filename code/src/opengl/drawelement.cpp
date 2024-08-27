#include <reclib/opengl/camera.h>
#include <reclib/opengl/drawelement.h>
#include <reclib/opengl/mesh_templates.h>

#include <iostream>

// -------------------------------------------------------
// DrawelementImpl
// -------------------------------------------------------

reclib::opengl::DrawelementImpl::DrawelementImpl(const std::string& name,
                                                 const Shader& shader,
                                                 const Mesh& mesh,
                                                 const mat4& model)
    : name(name),
      model(model),
      shader(shader),
      mesh(mesh),
      wireframe_mode(false),
      disable_render(false),
      is_grouped(false) {}

reclib::opengl::DrawelementImpl::~DrawelementImpl() {}

void reclib::opengl::DrawelementImpl::destroy_handles() {
  if (this->mesh.initialized()) {
    this->mesh->destroy_handles();
    if (reclib::opengl::Mesh::valid(mesh->name)) {
      auto elem = reclib::opengl::Mesh::find(mesh->name);
      this->mesh = reclib::opengl::Mesh();
      elem.free();
    }
  }
}

void reclib::opengl::DrawelementImpl::set_model_translation(const vec3& trans) {
  model.block<3, 1>(0, 3) = trans;
}
void reclib::opengl::DrawelementImpl::set_model_rotation(const mat3& rot) {
  model.block<3, 3>(0, 0) = rot;
}

void reclib::opengl::DrawelementImpl::bind() const {
  if (disable_render) return;

  if (shader) {
    shader->bind();
    if (mesh) mesh->bind(shader);
    shader->uniform("model", model);
    shader->uniform("model_normal", (mat4)transpose(inverse(model)));
    shader->uniform("view", current_camera()->view);
    shader->uniform("view_normal", current_camera()->view_normal);
    shader->uniform("proj", current_camera()->proj);
  }
}

void reclib::opengl::DrawelementImpl::unbind() const {
  if (disable_render) return;

  if (mesh) mesh->unbind();
  if (shader) shader->unbind();
}

void reclib::opengl::DrawelementImpl::set_wireframe_mode(bool mode) {
  wireframe_mode = mode;
}

void reclib::opengl::DrawelementImpl::set_disable_render(bool mode) {
  disable_render = mode;
}

void reclib::opengl::DrawelementImpl::draw() const {
  if (disable_render) return;

  for (auto iter = pre_draw_funcs.begin(); iter != pre_draw_funcs.end();
       iter++) {
    iter->second();
  }

  if (wireframe_mode) {
    draw_wireframe();
  } else {
    if (mesh) mesh->draw();
  }

  for (auto iter = post_draw_funcs.begin(); iter != post_draw_funcs.end();
       iter++) {
    iter->second();
  }
}

void reclib::opengl::DrawelementImpl::draw_bb() const {
  if (disable_render) return;

  if (reclib::opengl::Drawelement::valid(name + "_bb")) {
    reclib::opengl::Drawelement d =
        reclib::opengl::Drawelement::find(name + "_bb");
    d->bind();
    d->draw_wireframe();
    d->unbind();
  } else {
    vec3 bb_max = mesh->geometry->bb_max;
    vec3 bb_min = mesh->geometry->bb_min;

    if (!reclib::opengl::Material::valid("default_bb_material")) {
      reclib::opengl::Material m("default_bb_material");
      m->vec4_map["color"] = vec4(1, 1, 0, 1);
    }
    if (!reclib::opengl::Shader::valid("uniform_color")) {
      reclib::opengl::Shader("uniform_color", "MVP.vs", "color4Uniform.fs");
    }

    reclib::opengl::BoundingBox bb(
        name + "_bb", bb_min, bb_max,
        reclib::opengl::Material::find("default_bb_material"));

    reclib::opengl::Drawelement d(
        name + "_bb", reclib::opengl::Shader::find("uniform_color"), bb);
    d->model = model;

    d->bind();
    d->draw_wireframe();
    d->unbind();
  }
}

void reclib::opengl::DrawelementImpl::draw_wireframe() const {
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  if (mesh) mesh->draw();
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void reclib::opengl::DrawelementImpl::draw_index(uint32_t index) const {
  if (mesh) mesh->draw_index(index);
}

// -------------------------------------------------------
// GroupedDrawelementsImpl
// -------------------------------------------------------

reclib::opengl::GroupedDrawelementsImpl::GroupedDrawelementsImpl(
    const std::string& name,
    const std::vector<reclib::opengl::Drawelement>& elems)
    : name(name), elems(elems), wireframe_mode(false), disable_render(false) {
  for (unsigned int i = 0; i < elems.size(); i++) {
    this->elems[i]->is_grouped = true;
  }
}

reclib::opengl::GroupedDrawelementsImpl::~GroupedDrawelementsImpl() {}

void reclib::opengl::GroupedDrawelementsImpl::destroy_handles() {
  for (unsigned int i = 0; i < elems.size(); i++) {
    elems[i]->is_grouped = false;
    elems[i]->destroy_handles();
  }
}

void reclib::opengl::GroupedDrawelementsImpl::set_model_translation(
    const vec3& trans) {
  for (unsigned int i = 0; i < elems.size(); i++) {
    elems[i]->set_model_translation(trans);
  }
}
void reclib::opengl::GroupedDrawelementsImpl::set_model_rotation(
    const mat3& rot) {
  for (unsigned int i = 0; i < elems.size(); i++) {
    elems[i]->set_model_rotation(rot);
  }
}

void reclib::opengl::GroupedDrawelementsImpl::set_wireframe_mode(bool mode) {
  wireframe_mode = mode;
  for (unsigned int i = 0; i < elems.size(); i++) {
    elems[i]->set_wireframe_mode(mode);
  }
}

void reclib::opengl::GroupedDrawelementsImpl::set_disable_render(bool mode) {
  disable_render = mode;
  for (unsigned int i = 0; i < elems.size(); i++) {
    elems[i]->set_disable_render(mode);
  }
}

void reclib::opengl::GroupedDrawelementsImpl::bind_draw_unbind() const {
  if (disable_render) return;
  for (unsigned int i = 0; i < elems.size(); i++) {
    elems[i]->bind();
    elems[i]->draw();
    elems[i]->unbind();
  }
}

void reclib::opengl::GroupedDrawelementsImpl::bind_draw_bb_unbind() const {
  if (disable_render) return;
  for (unsigned int i = 0; i < elems.size(); i++) {
    elems[i]->bind();
    elems[i]->draw_bb();
    elems[i]->unbind();
  }
}

void reclib::opengl::GroupedDrawelementsImpl::bind_draw_wireframe_unbind()
    const {
  for (unsigned int i = 0; i < elems.size(); i++) {
    elems[i]->bind();
    elems[i]->draw_wireframe();
    elems[i]->unbind();
  }
}
