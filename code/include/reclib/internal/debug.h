#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>
#include <memory>

namespace reclib {
namespace debug {
void enable_stack_trace_on_crash();
void disable_stack_trace_on_crash();

void enable_gl_debug_output();
void disable_gl_debug_output();

void enable_gl_notifications();
void disable_gl_notifications();
}  // namespace debug

}  // namespace reclib

#endif
