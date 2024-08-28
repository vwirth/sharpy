# Works with 3.11 and tested through 3.20
cmake_minimum_required(VERSION 3.11...3.20)

if(NOT UNIX AND NOT WIN32)
	message(FATAL_ERROR "Unsupported platform!")
endif()

# #####################################################################
# ############################### OPTIONS #############################
# #####################################################################
option(WITH_VCPKG "Whether to use dependencies from vcpkg or search manually" OFF)
option(BUILD_EXAMPLES "Whether to build additional code examples" ON)
option(BUILD_TESTS "Whether to build additional tests" OFF)
option(INSTALL_MISSING_REQUIRED_DEPENDENCIES "Automatically download and install missing dependencies that are required" ON)
option(CUDA_ARCHITECTURE "The architecture name of your NVIDIA GPU, e.g. Pascal" "Ampere")
option(WINDOWS_ARCHITECTURE "Either 32Bit or 64Bit" "64Bit")
option(DEBUG_MODE "The debug mode enables assertions and further checks that would consume computation time otherwise" ON)
option(DEBUG_CMAKE_FLAGS "Whether to disable compiler optimization and instead insert debug flags" ON)

option(WITH_CUDA "Whether to build CUDA files or not. Warning: this en-/disables all functions that use CUDA." ON)
option(WITH_PYTHON "Whether to build code that invokes Python scripts or not. Warning: this en-/disables all functions that use Python." OFF)
option(OPENCV_MODULE "Whether to build OpenCV or not" ON)
option(CAMERA_MODULE "Whether to build camera module or not" OFF)
option(CERES_MODULE "Whether to build ceres modules or not" OFF)
option(OPENGL_MODULE "Whether to build opengl modules or not" ON)
option(DNN_MODULE "Whether to build deep learning modules or not" ON)
option(NVDIFFRAST_MODULE "Whether to build nvdiffrast (Differential Renderer) or not" ON)

option(VSMODE "Whether to build in Visual C++ in 'Debug' or 'Release' mode" "Release")

# place libraries and binaries despite their local file structure
# in the binary directory, e.g. build
option(OUTPUT_DIR "The output directory of the project binaries and libraries" "${CMAKE_CURRENT_BINARY_DIR}")

# for debugging
# set(VERBOSE True)
# set(CMAKE_VERBOSE_MAKEFILE ON)

# #####################################################################
# ############################ VARIABLES ##############################
# #####################################################################
set(vcpkg "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

if(NOT CMAKE_TOOLCHAIN_FILE AND EXISTS "${vcpkg}" AND WITH_VCPKG)
	set(CMAKE_TOOLCHAIN_FILE "${vcpkg}"
		CACHE FILEPATH "CMake toolchain file")
	message(STATUS "vcpkg toolchain found: ${CMAKE_TOOLCHAIN_FILE}")
endif()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(PROJECT_LANGUAGES CXX CUDA)

# set(PROJECT_LANGUAGES CXX)
set(PROJECT_NAME reconstruction-lib)

set(WINDOWS_ARCHITECTURE "64Bit") # Either 32Bit or 64Bit
set(CMAKE_CUDA_ARCHITECTURES 86) # compute capability of GPU

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(GenerateExportHeader)

# ############################ OUTPUT ##############################

# Make to cache variable so it can be overwritten by the user
set(PROJECT_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}/bin" CACHE PATH "Output dir of executables and dynamic libraries")
set(PROJECT_LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}/lib" CACHE PATH "Output dir of static libraries")
set(PROJECT_ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}/lib" CACHE PATH "Output dir of static libraries")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_LIBRARY_OUTPUT_DIRECTORY}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_ARCHIVE_OUTPUT_DIRECTORY}")

# #####################################################################
# ######################### GENERIC STUFF #############################
# #####################################################################

# Project name and a few useful settings. Other commands can pick up the results
project(
	${PROJECT_NAME}
	VERSION 0.1
	DESCRIPTION "Reconstruction Library"
	LANGUAGES ${PROJECT_LANGUAGES}
)

# #####################################################################
# ######################## PROJECT DEPENDENCIES #######################
# #####################################################################
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(helper_macros)
include(Dependencies_core)

if(WITH_PYTHON)
	find_package(
		Python3
		REQUIRED
		COMPONENTS Interpreter
	)
endif()

if(WITH_CUDA)
	# TODO: pass via env?
	set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
	list(APPEND CMAKE_PREFIX_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64/")

	# # CUDA
	find_package(CUDA 11.4 REQUIRED)
	set(CUDA_LIBRARIES ${CUDA_LIBRARIES} cuda)
	PackageHelper(CUDA "${CUDA_FOUND}" "${CUDA_INCLUDE_DIRS}" "${CUDA_LIBRARIES}")
endif()

if(OPENCV_MODULE)
	include(Dependencies_opencv)
endif()

if(OPENGL_MODULE)
	include(Dependencies_opengl)
endif()

if(CAMERA_MODULE)
	include(Dependencies_camera)
endif()

if(DNN_MODULE)
	include(Dependencies_dnn)
endif()

if(NVDIFFRAST_MODULE)
	include(Dependencies_nvdiffrast)
endif()

IF(CUDA_FOUND)
	include_directories("${CUDA_INCLUDE_DIRS}")
ENDIF()

# #####################################################################
# ######################### COMPILER STUFF ############################
# #####################################################################

# Optionally set things like CMAKE_CXX_STANDARD, CMAKE_POSITION_INDEPENDENT_CODE here

# Let's ensure -std=g++xx instead of -std=c++xx
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_EXTENSIONS ON)

if(NOT DEFINED CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Set C++ standard required for CUDA
if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 14)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Compiler options
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(CCACHE_FOUND ccache)

if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

if(UNIX)
	# Compiler flags:
	# -03 											More aggressive than -O2 with longer compile times.
	# Recommended for codes that loops involving intensive floating point calculations.
	# -funroll-loops 						Unroll loops whose number of iterations can be determined at compile time or upon entry to the loop
	# -g												Generate debugging information
	# -pthread									Enable multithreading
	# -Wall											Enables warnings, must be enabled before compiler flags that disable some warnings again
	# -Wno-unused-parameter			Disables warnings for unused parameters
	# -Wno-unused-variable			Disables warnings for unused variables
	# -Wno-int-to-pointer-cast 	Suppress warnings from casts to pointer type of an integer of a different size.
	# -march=native							Allows use of instructions that require the listed architecture.
	# if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
	# else()
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
	# endif()
	if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		# if you want to have .profraw files use this flag: -fprofile-generate
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v -fvectorize")

		# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_86")
	endif()

	if(NOT DEBUG_CMAKE_FLAGS)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -G")
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
	endif()

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -pthread -march=native -finline-functions")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-parameter -Wno-unused-variable -Wno-int-to-pointer-cast -Wno-deprecated-declarations")

	# linker flags
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")

	# Use the following lines if using the plain lld, not a specific version defined in settings.json
	# find_program(LLD_FOUND lld)

	# if(LLD_FOUND)
	# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
	# else()
	# message(WARNING "Install `lld` to make linking significantly faster.")
	# endif()
	if(DNN_MODULE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -frecord-gcc-switches")
	endif()

	# use the flags -g and -G with caution when it comes to CUDA code
	# -Xptxas
	# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -Wno-deprecated-declarations -v -gencode arch=compute_86,code=compute_86 -gencode arch=compute_61,code=compute_61 -Xcompiler -Wno-cpp --expt-relaxed-constexpr")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -Wno-deprecated-declarations -v -gencode arch=compute_86,code=compute_86 -gencode arch=compute_61,code=compute_61 -Xcompiler -Wno-cpp")

	# position independent code is required for shared libs. For GCC this is -fPIC
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
else()
	# Windows stuff
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -D_USE_MATH_DEFINES -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING /bigobj")
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

# #####################################################################
# ######################## ADDITIONAL STUFF ###########################
# #####################################################################

# Let's nicely support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# #####################################################################
# ######################## ADD SUBDIRECTORIES ####################
# #####################################################################

# The compiled library code is here
add_subdirectory(src)

if(BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

# #####################################################################
# ######################## VERBOSE OUTPUT ############################
# #####################################################################
message(STATUS "\nCompiler Flags:")

message(STATUS "PROJECT_CXX_FLAGS: ${PROJECT_CXX_FLAGS}")
message(STATUS "PROJECT_PRIVATE_CXX_FLAGS: ${PROJECT_PRIVATE_CXX_FLAGS}")
message(STATUS "PROJECT_LD_FLAGS: ${PROJECT_LD_FLAGS}")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")

if(CUDA_FOUND)
	message(STATUS "\nCUDA Compiler Flags:")
	message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
	message(STATUS "CMAKE_CUDA_FLAGS_DEBUG: ${CMAKE_CUDA_FLAGS_DEBUG}")
	message(STATUS "CMAKE_CUDA_FLAGS_RELWITHDEBINFO: ${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
	message(STATUS "CMAKE_CUDA_FLAGS_RELEASE: ${CMAKE_CUDA_FLAGS_RELEASE}")
	message(STATUS "CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")
endif()
