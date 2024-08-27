# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.

# Optionally glob, but only for CMake 3.12 or later:
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

if(WITH_CUDA)
  file(GLOB_RECURSE SOURCE_LIST *.cpp *.cu)
  file(GLOB_RECURSE HEADER_LIST
    "${INCLUDE_DIR}/*.h"
    "${INCLUDE_DIR}/*.hpp"
    "${INCLUDE_DIR}/*.cuh"
  )
else()
  file(GLOB_RECURSE SOURCE_LIST *.cpp)
  file(GLOB_RECURSE HEADER_LIST
    "${INCLUDE_DIR}/*.h"
    "${INCLUDE_DIR}/*.hpp"
  )
endif()

set(CUSTOM_BINARIES
)

# #####################################################################
# ############################ VARIABLES ##############################
# #####################################################################
foreach(SRC ${SOURCE_LIST})
  string(REPLACE ".cpp" "" BIN ${SRC})
  string(REGEX REPLACE ".*/examples/" "" BIN ${BIN})
  set(BINARY_NAMES ${BINARY_NAMES} ${BIN})
endforeach()

# set(BINARY_DEPENDENCIES
# ${CUDA_LIBRARIES}
# reclib
# )
set(BINARY_DEPENDENCIES

  # ${CUDA_LIBRARIES}
  # ${OPENCV_LIBS} ${OPENCV_TARGETS}
  # ${CORE_LIBS} ${CORE_TARGETS}
  # ${CAMERA_LIBS} ${CAMERA_TARGETS}
  # ${CERES_TARGETS} ${CERES_LIBS}
  # ${OPENGL_TARGETS} ${OPENGL_LIBS}
  # ${DNN_TARGETS} ${DNN_LIBS}
  reclib

  # $<TARGET_OBJECTS:reclib>
)

set(BINARY_INCLUDES
  ${CUDA_INCLUDE_DIRS}
  ${OPENCV_INCLUDES}
  ${CORE_INCLUDES}
  ${CAMERA_INCLUDES}
  ${CERES_INCLUDES}
  ${OPENGL_INCLUDES}
  ${DNN_INCLUDES}
)

if(WITH_PYTHON)
  set(BINARY_INCLUDES ${BINARY_INCLUDES} ${Python3_INCLUDE_DIRS})
endif()

# #####################################################################
# ############### ADDITIONAL DEPENDENCIES #############################
# #####################################################################
# FetchContent added in CMake 3.11, downloads during the configure step
# include(FetchContent)
# include(ExternalProject)

# Formatting library
# FetchContent_Declare(
# fmtlib
# GIT_REPOSITORY https://github.com/fmtlib/fmt.git
# GIT_TAG 5.3.0)
# FetchContent_MakeAvailable(fmtlib)

# #####################################################################
# ######################## BUILD BINARIES #############################
# #####################################################################
foreach(BINARY ${BINARY_NAMES})
  if(${BINARY} IN_LIST CUSTOM_BINARIES)
    message(STATUS "Skipping binary '${BINARY}' in automatic compilation")
    continue()
  endif()

  add_executable(${BINARY} ${BINARY}.cpp)

  target_include_directories(${BINARY} PRIVATE
    ${INCLUDE_DIR} ${BINARY_INCLUDES}
  )

  # Optional for libraries
  target_link_libraries(${BINARY} PUBLIC
    ${BINARY_DEPENDENCIES}
  )

  target_compile_definitions(${BINARY} PUBLIC EXAMPLE_DIRECTORY_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()

# message("HEYY ${BINARY_NAMES}")
# add_executable(clang_bug "clang_bug.cpp")
# target_include_directories(clang_bug PRIVATE ${INCLUDE_DIR} ${BINARY_INCLUDES})
# target_link_libraries(clang_bug PUBLIC reclib)

# #####################################################################
# ######################## CUSTOM BINARIES ###########################
# #####################################################################

# ############################### MANO2RGBD #########################
# list(GET CUSTOM_BINARIES 0 BINARY)
# message(STATUS "Manually compiling '${BINARY}'...")

# file(GLOB_RECURSE SOURCE_LIST *.cpp)

# set(SRC
# "${BINARY}.cpp"
# )
# set(INCLUDE
# "${INCLUDE_DIR}"
# )
# add_executable(${BINARY} ${BINARY}.cpp)

# target_include_directories(${BINARY} PRIVATE
# ${INCLUDE}
# ${CUDA_INCLUDE_DIRS}
# ${OPENCV_INCLUDES}
# ${CORE_INCLUDES}
# ${CAMERA_INCLUDES}
# ${CERES_INCLUDES}
# ${OPENGL_INCLUDES}
# ${DNN_INCLUDES}
# )

# # Optional for libraries
# target_link_libraries(${BINARY} PUBLIC
# ${CORE_LIBS} ${CORE_TARGETS}
# ${CUDA_LIBRARIES}
# ${DNN_TARGETS} ${DNN_LIBS} # for some magical reason, the DNN targets have to be linked before all other targets, otherwise torch::jit::load will not work
# ${OPENCV_LIBS} ${OPENCV_TARGETS}
# ${CAMERA_LIBS} ${CAMERA_TARGETS}
# ${CERES_TARGETS} ${CERES_LIBS}
# ${OPENGL_TARGETS} ${OPENGL_LIBS}
# )

# target_compile_definitions(${BINARY} PUBLIC EXAMPLE_DIRECTORY_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

# #####################################################################
# ######################## ADDITIONAL STUFF ###########################
# #####################################################################
# IDEs should put the headers in a nice place
source_group(
  TREE "${INCLUDE_DIR}"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
source_group(
  TREE "${CMAKE_CURRENT_SOURCE_DIR}"
  PREFIX "Source Files"
  FILES ${SOURCE_LIST}
)
