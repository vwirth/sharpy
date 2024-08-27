#ifndef RECLIB_PYTHON_PYTHON_H
#define RECLIB_PYTHON_PYTHON_H

#include "reclib/python/npy2eigen.h"

#if WITH_PYTHON

#include <Python.h>

#include <string>
#include <vector>

#include "reclib/internal/filesystem.h"

namespace reclib {
namespace python {
void execute_func_noargs(const fs::path& filename,
                         const std::string& func_name);

void execute_func_stringargs(const fs::path& filename,
                             const std::string& func_name,
                             const std::vector<std::string>& args);
}  // namespace python
}  // namespace reclib

#endif

#endif