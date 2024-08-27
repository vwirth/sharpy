# Defines the following output variables:
#
# NVDIFFRAST_INCLUDES:    The list of required include directories
# NVDIFFRAST_LIBS:        The list of required libraries for link_target
# NVDIFFRAST_TARGETS:     The list of required targets
# MODULE_NVDIFFRAST:      True if all required dependencies are found.
#

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_DNN)

set(DEBUG_INSTALLATION ON)
set(FORCE_MANUAL_INSTALLATION OFF)
set(BUILD_SHARED_LIBS ON)

include(FetchContent)

# allow no updates -> faster
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# # -----------------------------------------------------
# # -----------------------------------------------------

set(Nvdiffrast_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/nvdiffrast")

IncludeHelper(Nvdiffrast_INCLUDE_DIRS nvdiffrast/common/common.h "${Nvdiffrast_BASE_PATH}" Nvdiffrast_FOUND)

set(Nvdiffrast_DIR "${Nvdiffrast_BASE_PATH}")

if(WITH_CUDA)
	file(GLOB_RECURSE Nvdiffrast_LIBRARIES "${Nvdiffrast_DIR}/*.cpp" "${Nvdiffrast_DIR}/*.cu")
else()
	file(GLOB_RECURSE Nvdiffrast_LIBRARIES "${Nvdiffrast_DIR}/*.cpp" "${Nvdiffrast_DIR}/*.cu")
endif()

PackageHelper(Nvdiffrast "${Nvdiffrast_FOUND}" "${Nvdiffrast_INCLUDE_DIRS}" "${Nvdiffrast_LIBRARIES}")
unset(Nvdiffrast_BASE_PATH)


unset(DEBUG_INSTALLATION)
unset(BUILD_SHARED_LIBS)

set(NVDIFFRAST_INCLUDES ${PACKAGE_INCLUDES})
set(NVDIFFRAST_LIBS ${LIBS})
set(NVDIFFRAST_TARGETS ${LIB_TARGETS})
set(MODULE_NVDIFFRAST 1)
