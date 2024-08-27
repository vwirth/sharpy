#ifndef RECLIB_OPENGL_QUAD_H
#define RECLIB_OPENGL_QUAD_H

// clang-format off
#include <GL/glew.h>
#include <GL/gl.h>
// clang-format on

// ----------------------------
// Definition
namespace reclib {
namespace opengl {

class _API Quad {
 public:
  static inline void draw();
  inline void draw_obj();
  inline Quad(vec2 offset = vec2(0, 0), vec2 scale = vec2(1, 1),
              bool reverse_y = false);
  inline virtual ~Quad();

 private:
  inline void draw_internal() const;
  GLuint vao, vbo, ibo;
};

// ----------------------------
// Implementation

Quad::Quad(vec2 offset, vec2 scale, bool reverse_y) {
  float quad[20] = {0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1};
  if (reverse_y) {
    for (unsigned int i = 4; i < 20; i = i + 5) {
      if (quad[i] == 0) {
        quad[i] = 1;
      } else if (quad[i] == 1) {
        quad[i] = 0;
      }
    }
  }

  for (unsigned int i = 0; i < 20; i = i + 5) {
    quad[i] = quad[i] * scale.x() + offset.x();
    quad[i + 1] = quad[i + 1] * scale.y() + offset.y();
  }

  static uint32_t idx[6] = {0, 1, 2, 2, 3, 0};

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
  glGenBuffers(1, &ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, 0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5,
                        (GLvoid*)(sizeof(float) * 3));
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

Quad::~Quad() {
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &ibo);
  glDeleteBuffers(1, &vbo);
}

void Quad::draw() {
  static Quad quad;
  quad.draw_internal();
}
void Quad::draw_obj() { draw_internal(); }

void Quad::draw_internal() const {
  glBindVertexArray(vao);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);
}

}  // namespace opengl
}  // namespace reclib

#endif
