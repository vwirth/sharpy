# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.

# Optionally glob, but only for CMake 3.12 or later:
set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(DEP_DIR "${PROJECT_SOURCE_DIR}/external")

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

if(OPENGL_MODULE)
  file(GLOB IMGUI_SOURCE "${PROJECT_SOURCE_DIR}/external/thirdparty/include/imgui/*.cpp")
  set(IMGUI_SOURCE ${IMGUI_SOURCE}
    "${PROJECT_SOURCE_DIR}/external/thirdparty/include/imgui/backends/imgui_impl_glfw.cpp"
    "${PROJECT_SOURCE_DIR}/external/thirdparty/include/imgui/backends/imgui_impl_opengl3.cpp")
endif()

# #####################################################################
# ############################ VARIABLES ##############################
# #####################################################################
set(LIB_NAME reclib)
set(LIB_DEPENDENCIES
  ${CORE_LIBS} ${CORE_TARGETS}
  ${CUDA_LIBRARIES}
  ${DNN_TARGETS} ${DNN_LIBS} # for some magical reason, the DNN targets have to be linked before all other targets, otherwise torch::jit::load will not work
  ${OPENCV_LIBS} ${OPENCV_TARGETS}
  ${CAMERA_LIBS} ${CAMERA_TARGETS}
  ${CERES_TARGETS} ${CERES_LIBS}
  ${OPENGL_TARGETS} ${OPENGL_LIBS}
)

if(UNIX)
  # set(LIB_DEPENDENCIES ${LIB_DEPENDENCIES} std++fs)
endif()

# additional include directories that are not already satisfied by
# the targets in LIB_DEPENDENCIES
set(LIB_INCLUDES
  ${CUDA_INCLUDE_DIRS}
  ${OPENCV_INCLUDES}
  ${CORE_INCLUDES}
  ${CAMERA_INCLUDES}
  ${CERES_INCLUDES}
  ${OPENGL_INCLUDES}
  ${DNN_INCLUDES}
  ${NVDIFFRAST_INCLUDES}
)

set(BINARY_NAME main)
set(BINARY_DEPENDENCIES ${LIB_NAME} ${LIB_DEPENDENCIES})

set(EXTENDED_HEADER_LIST ${HEADER_LIST})
set(EXTENDED_SOURCE_LIST ${SOURCE_LIST})

if(NVDIFFRAST_MODULE)
  set(EXTENDED_HEADER_LIST ${EXTENDED_HEADER_LIST} ${NVDIFFRAST_INCLUDES})
  set(EXTENDED_SOURCE_LIST ${EXTENDED_SOURCE_LIST} ${NVDIFFRAST_TARGETS} ${NVDIFFRAST_LIBS})
endif()

# #####################################################################
# ############### ADDITIONAL DEPENDENCIES #############################
# #####################################################################

# #####################################################################
# ############### BUILD OWN STATIC LIBRARY FROM INCLUDE FILES #########
# #####################################################################
add_library(${LIB_NAME} OBJECT ${EXTENDED_SOURCE_LIST} ${IMGUI_SOURCE} ${EXTENDED_HEADER_LIST})

# add_library(${LIB_NAME} SHARED ${SOURCE_LIST} ${IMGUI_SOURCE} ${HEADER_LIST})

# generate_export_header(${LIB_NAME})
set_target_properties(${LIB_NAME} PROPERTIES VERSION 1.0 DESCRIPTION "xx")

target_compile_definitions(${LIB_NAME} PUBLIC RECLIB_LIBRARY_SOURCE_PATH="${PROJECT_SOURCE_DIR}")
target_compile_definitions(${LIB_NAME} PUBLIC RECLIB_DATA_DIRECTORY="${PROJECT_SOURCE_DIR}/../data")
target_compile_definitions(${LIB_NAME} PUBLIC RECLIB_SHADER_DIRECTORY="${PROJECT_SOURCE_DIR}/shaders")

if(DEBUG_MODE)
  target_compile_definitions(${LIB_NAME} PUBLIC RECLIB_DEBUG_MODE=1)
endif()

if(WITH_CUDA)
  target_compile_definitions(${LIB_NAME} PUBLIC WITH_CUDA)
endif()

if(WITH_PYTHON)
  target_compile_definitions(${LIB_NAME} PUBLIC WITH_PYTHON)
endif()

if(OPENCV_MODULE)
  target_compile_definitions(${LIB_NAME} PUBLIC HAS_OPENCV_MODULE)
endif()

if(OPENGL_MODULE)
  target_compile_definitions(${LIB_NAME} PUBLIC HAS_OPENGL_MODULE)
endif()

if(CAMERA_MODULE)
  target_compile_definitions(${LIB_NAME} PUBLIC HAS_CAMERA_MODULE)
endif()

if(CERES_MODULE)
  target_compile_definitions(${LIB_NAME} PUBLIC HAS_CERES_MODULE)
endif()

if(DNN_MODULE)
  target_compile_definitions(${LIB_NAME} PUBLIC HAS_DNN_MODULE)
endif()

if(NVDIFFRAST_MODULE)
  target_compile_definitions(${LIB_NAME} PUBLIC HAS_NVDIFFRAST_MODULE)
  target_compile_definitions(${LIB_NAME} PUBLIC NVDR_TORCH)
endif()

if(freenect2_FOUND)
  target_compile_definitions(${LIB_NAME} PUBLIC HAS_FREENECT2)
endif(freenect2_FOUND)

if(k4a_FOUND)
  target_compile_definitions(${LIB_NAME} PUBLIC HAS_K4A)
endif(k4a_FOUND)

# add the linker flags, compile flags and include paths of all
# dependencies to the current library via one unified command
target_link_libraries(${LIB_NAME} PUBLIC ${LIB_DEPENDENCIES})

# set visibility to PUBLIC such that linked binaries can use the
# include files
target_include_directories(${LIB_NAME} PUBLIC ${INCLUDE_DIR})

# add additional include directories that are not contained in the
# targets of LIB_DEPENDENCIES
target_include_directories(${LIB_NAME} PUBLIC ${LIB_INCLUDES})

# #####################################################################
# ######################## BUILD BINARIES #############################
# #####################################################################

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
