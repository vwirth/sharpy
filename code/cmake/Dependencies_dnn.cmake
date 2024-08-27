# Defines the following output variables:
#
# DNN_INCLUDES:    The list of required include directories
# DNN_LIBS:        The list of required libraries for link_target
# DNN_TARGETS:     The list of required targets
# MODULE_DNN:      True if all required dependencies are found.
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

# # CUDA

# set(CUDA_LIBRARIES ${CUDA_LIBRARIES} cuda cublas)
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} cuda)
PackageHelper(CUDA "${CUDA_FOUND}" "${CUDA_INCLUDE_DIRS}" "${CUDA_LIBRARIES}")

# # -----------------------------------------------------
# # -----------------------------------------------------

# Torch 1.13 with cuda 11.8
set(Torch_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/Torch")

# LibraryHelper(TORCH_LIBRARIES_DEPS libtorch_global_deps.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)
find_package(Torch QUIET HINTS "${Torch_BASE_PATH}/source/share/cmake/Torch")

# set(TORCH_LIBRARIES ${TORCH_LIBRARIES_DEPS} ${TORCH_LIBRARIES} )
# LibraryHelper(TORCH_LIBRARIES libtorch.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)
# LibraryHelper(TORCH_LIBRARIES libc10.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)
# LibraryHelper(TORCH_LIBRARIES libc10_cuda.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)
# LibraryHelper(TORCH_LIBRARIES libtorch_cpu.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)
# LibraryHelper(TORCH_LIBRARIES libtorch_cuda.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)
# LibraryHelper(TORCH_LIBRARIES libtorch_cuda_cpp.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)
# LibraryHelper(TORCH_LIBRARIES libtorch_cuda_cu.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)
# IncludeHelper(TORCH_INCLUDE_DIRS "torch/torch.h" "${Torch_BASE_PATH}/source/include/torch/csrc/api/include" Torch_FOUND)
# IncludeHelper(TORCH_INCLUDE_DIRS "torch/script.h" "${Torch_BASE_PATH}/source/include/" Torch_FOUND)

# LibraryHelper(TORCH_LIBRARIES libtorchbind_test.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)

# download and install manually
if((NOT Torch_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(Torch_FOUND)

	if(NOT Torch_POPULATED)
		message("Manually download, configure and build Torch library...")
		FetchContent_Declare(
			Torch

			URL "https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip"

			# URL "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${Torch_BASE_PATH}/install"
			PREFIX "${Torch_BASE_PATH}"
			DOWNLOAD_DIR "${Torch_BASE_PATH}/download"
			SOURCE_DIR "${Torch_BASE_PATH}/source"
			BINARY_DIR "${Torch_BASE_PATH}/build"
			SUBBUILD_DIR "${Torch_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(Torch)
	endif()

	set(Torch_POPULATED ${Torch_POPULATED} CACHE BOOL "Whether Torch lib was manually loaded and installed" FORCE)

	set(Torch_DIR "${Torch_BASE_PATH}/source/share/cmake/Torch")

	# find_package(Torch QUIET REQUIRED PATHS ${Torch_DIR})
	# since the Torch cmake script is incomplete we have to add some libraries manually
	LibraryHelper(TORCH_LIBRARIES libtorch_global_deps.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)
	LibraryHelper(TORCH_LIBRARIES libtorchbind_test.so "${Torch_BASE_PATH}/source/lib" Torch_FOUND)

	if(NOT Torch_FOUND)
		message(FATAL_ERROR "Could not find Torch.")
	endif()
endif()

PackageHelper(Torch "${Torch_FOUND}" "${TORCH_INCLUDE_DIRS}" "${TORCH_LIBRARIES}")

# PackageHelperTarget(torch Torch_FOUND)

# # -----------------------------------------------------
# # -----------------------------------------------------

# TensorRt 8.4.2-1 with cuda 11.6
if(UNIX)
	set(TensorRT_BASE_PATH "") # system-wide
elseif(WIN32)
	message(FATAL_ERROR "Win32 TensorRT not configured. Please check the CMakeLists.txt file and specify the required files.")
endif()

if(UNIX)
	LibraryHelper(TensorRT_LIBRARIES libnvinfer_plugin.so "${TensorRT_BASE_PATH}/lib" TensorRT_FOUND)
	LibraryHelper(TensorRT_LIBRARIES libnvparsers.so "${TensorRT_BASE_PATH}/lib" TensorRT_FOUND)
	LibraryHelper(TensorRT_LIBRARIES libnvinfer.so "${TensorRT_BASE_PATH}/lib" TensorRT_FOUND)
	message("TENSORRRRRRT: ${TensorRT_LIBRARIES}")
elseif(WIN32)
	message(FATAL_ERROR "Win32 Torch-TensorRT not configured. Please check the CMakeLists.txt file and specify the required files.")
endif()

IncludeHelper(TensorRT_INCLUDE_DIRS "NvInfer.h" "${TensorRT_BASE_PATH}/include" TensorRT_FOUND)

message(STATUS "Found: ${TensorRT_FOUND}")

if(NOT TensorRT_FOUND)
	message(FATAL_ERROR "Could not find TensorRT. Please make sure to install it, see https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html")
endif()

PackageHelper(TensorRT "${TensorRT_FOUND}" "${TensorRT_INCLUDE_DIRS}" "${TensorRT_LIBRARIES}")

# # -----------------------------------------------------
# # -----------------------------------------------------

# Torch_TensorRT 1.11 with cuda 11.3
set(Torch_TensorRT_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/torch_tensorrt")

if(UNIX)
	LibraryHelper(Torch_TensorRT_LIBRARIES libtorchtrt.so "${Torch_TensorRT_BASE_PATH}/source/lib" Torch_TensorRT_FOUND)

# LibraryHelper(Torch_TensorRT_LIBRARIES libtorchtrt_runtime.so "${Torch_BASE_PATH}/source/lib" Torch_TensorRT_FOUND)
# LibraryHelper(Torch_TensorRT_LIBRARIES libtorchtrt_plugins.so "${Torch_BASE_PATH}/source/lib" Torch_TensorRT_FOUND)
elseif(WIN32)
	message(FATAL_ERROR "Win32 Torch-TensorRT not configured. Please check the CMakeLists.txt file and specify the required files.")
endif()

IncludeHelper(Torch_TensorRT_INCLUDE_DIRS torch_tensorrt/torch_tensorrt.h "${Torch_TensorRT_BASE_PATH}/source/include" Torch_TensorRT_FOUND)
IncludeHelper(Torch_TensorRT_INCLUDE_DIRS core/compiler.h "${Torch_TensorRT_BASE_PATH}/source/include/torch_tensorrt" Torch_TensorRT_FOUND)

# download and install manually
if((NOT Torch_TensorRT_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(Torch_TensorRT_FOUND)

	if(NOT Torch_TensorRT_POPULATED)
		message("Manually download, configure and build Torch_TensorRT library...")
		FetchContent_Declare(
			Torch_TensorRT
			URL "https://github.com/pytorch/TensorRT/releases/download/v1.1.0/libtorchtrt-v1.1.0-cudnn8.2-tensorrt8.2-cuda11.3-libtorch1.11.0.tar.gz"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${Torch_TensorRT_BASE_PATH}/install"
			PREFIX "${Torch_TensorRT_BASE_PATH}"
			DOWNLOAD_DIR "${Torch_TensorRT_BASE_PATH}/download"
			SOURCE_DIR "${Torch_TensorRT_BASE_PATH}/source"
			BINARY_DIR "${Torch_TensorRT_BASE_PATH}/build"
			SUBBUILD_DIR "${Torch_TensorRT_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(Torch_TensorRT)
	endif()

	set(Torch_TensorRT_POPULATED ${Torch_TensorRT_POPULATED} CACHE BOOL "Whether Torch_TensorRT lib was manually loaded and installed" FORCE)

	set(Torch_TensorRT_DIR "${Torch_TensorRT_BASE_PATH}/source/share/cmake/Torch_TensorRT")

	if(UNIX)
		# LibraryHelper(Torch_TensorRT_LIBTORCHTRT libtorchtrt.so "${Torch_TensorRT_BASE_PATH}/source/lib" Torch_TensorRT_FOUND)
		# file(COPY "${Torch_TensorRT_LIBTORCHTRT}" DESTINATION "${Torch_BASE_PATH}/source/lib")
		# LibraryHelper(Torch_TensorRT_LIBRARIES libtorchtrt.so "${Torch_BASE_PATH}/source/lib" Torch_TensorRT_FOUND)
		LibraryHelper(Torch_TensorRT_LIBRARIES libtorchtrt.so "${Torch_TensorRT_BASE_PATH}/source/lib" Torch_TensorRT_FOUND)

		# BEWARE: libtorchtrt.so is a bitch and has a dependency to libtorch_global_deps.so from Torch.
		# However, since the library is not included in Torch-TensorRT but instead in Torch
		# cmake does not find this dependency automatically and the c++ loader will not be able to find 'libtorch_global_deps.so'
		# at runtime
		#
		# The reason is that when compiling an executable, the runpath spec in the ELF format only points to directories
		# of DIRECT dependencies. However, 'libtorch_global_deps.so' is only a secondary dependency since only Torch-TensorRT
		# is depending on it and not the executably itself.
		execute_process(COMMAND bash "-c" "readelf -d ${Torch_TensorRT_BASE_PATH}/source/lib/libtorchtrt.so | grep RUNPATH" RESULT_VARIABLE RES OUTPUT_VARIABLE OUTPUT_RUNPATH ERROR_VARIABLE OUTPUT_ERROR)

		if(RES EQUAL 0)
			string(FIND ${OUTPUT_RUNPATH} "[" START_INDEX)
			string(FIND ${OUTPUT_RUNPATH} "]" END_INDEX)
			math(EXPR START_INDEX "${START_INDEX}+1")
			math(EXPR LENGTH "${END_INDEX}-${START_INDEX}")
			message("Patching runpath of libtorchtrt.so ....")
			string(SUBSTRING ${OUTPUT_RUNPATH} ${START_INDEX} ${LENGTH} TENSORRT_RUNPATH)

			unset(START_INDEX)
			unset(END_INDEX)
			unset(LENGTH)
			unset(RES)
			unset(OUTPUT_RUNPATH)
			unset(OUTPUT_ERROR)

			execute_process(COMMAND bash "-c" "patchelf --set-rpath ${TENSORRT_RUNPATH}:${Torch_BASE_PATH}/source/lib ${Torch_TensorRT_BASE_PATH}/source/lib/libtorchtrt.so" RESULT_VARIABLE RES OUTPUT_VARIABLE OUT ERROR_VARIABLE ERROR)

			if(RES EQUAL 0)
				message(STATUS "Patched Runpath of libtorchtrt.so!")
			else()
				message(FATAL_ERROR "Failed to patch libtorchtrt.so: ${ERROR}")
			endif()

			unset(ERROR)
			unset(OUT)
			unset(RES)
		else()
			message(FATAL_ERROR "Error: Tried to get RUNPATH variable from libtorchtrt.so. Received Error: ${OUTPUT_ERROR}")
		endif()

	elseif(WIN32)
		message(FATAL_ERROR "Win32 Torch-TensorRT not configured. Please check the CMakeLists.txt file and specify the required files.")
	endif()

	IncludeHelper(Torch_TensorRT_INCLUDE_DIRS torch_tensorrt/torch_tensorrt.h "${Torch_TensorRT_BASE_PATH}/source/include" Torch_TensorRT_FOUND)
	IncludeHelper(Torch_TensorRT_INCLUDE_DIRS core/compiler.h "${Torch_TensorRT_BASE_PATH}/source/include/torch_tensorrt" Torch_TensorRT_FOUND)

	if(NOT Torch_TensorRT_FOUND)
		message(FATAL_ERROR "Could not find Torch-TensorRT.")
	endif()

	unset(Torch_TensorRT_BASE_PATH)
endif()

PackageHelper(Torch_TensorRT "${Torch_TensorRT_FOUND}" "${Torch_TensorRT_INCLUDE_DIRS}" "${Torch_TensorRT_LIBRARIES}")
unset(Torch_BASE_PATH)

unset(DEBUG_INSTALLATION)
unset(BUILD_SHARED_LIBS)

set(DNN_INCLUDES ${PACKAGE_INCLUDES})
set(DNN_LIBS ${LIBS})
set(DNN_TARGETS ${LIB_TARGETS})
set(MODULE_DNN 1)
