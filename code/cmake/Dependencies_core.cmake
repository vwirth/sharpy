# Defines the following output variables:
#
# CORE_INCLUDES:    The list of required include directories
# CORE_LIBS:        The list of required libraries for link_target
# CORE_TARGETS:     The list of required targets
# MODULE_CORE:      True if all required dependencies are found.
#

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_CORE)

set(DEBUG_INSTALLATION ON)
set(FORCE_MANUAL_INSTALLATION OFF)
set(BUILD_SHARED_LIBS ON)

include(FetchContent)

# allow no updates -> faster
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Git submodule auto update
# https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git QUIET)

if(Git_FOUND AND NOT SUBMODULES_INITIALIZED)
	# Update submodules as needed
	option(GIT_SUBMODULE "Check submodules during build" ON)

	if(GIT_SUBMODULE)
		message(STATUS "Running update --init --recursive")
		execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
			RESULT_VARIABLE GIT_SUBMOD_RESULT)

		if(NOT GIT_SUBMOD_RESULT EQUAL "0")
			message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
		endif()

		message(STATUS "Submodule update done")
		set(SUBMODULES_INITIALIZED True CACHE BOOL "Whether submodules are initialized" FORCE)
	endif()
endif()

# # -----------------------------------------------------
# # -----------------------------------------------------

# # ALL HEADER-ONLY LIBRARIES in external/include
set(PACKAGE_INCLUDES ${PACKAGE_INCLUDES} "${PROJECT_SOURCE_DIR}/external/include")
set(PACKAGE_INCLUDES ${PACKAGE_INCLUDES} "${PROJECT_SOURCE_DIR}/external/thirdparty/include")

# # -----------------------------------------------------
# # -----------------------------------------------------

# # Eigen3
set(Eigen3_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/eigen3")
find_package(Eigen3 3.4.0 QUIET HINTS "${Eigen3_BASE_PATH}/build")

# download and install manually
if((NOT EIGEN3_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(EIGEN3_FOUND)

	if(NOT eigen3_POPULATED)
		message("Manually download, configure and build Eigen3 library...")
		FetchContent_Declare(
			Eigen3
			GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
			GIT_TAG "origin/3.4"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${Eigen3_BASE_PATH}"
			PREFIX "${Eigen3_BASE_PATH}"
			DOWNLOAD_DIR "${Eigen3_BASE_PATH}/download"
			SOURCE_DIR "${Eigen3_BASE_PATH}/source"
			BINARY_DIR "${Eigen3_BASE_PATH}/build"
			SUBBUILD_DIR "${Eigen3_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(Eigen3)
	endif()

	set(eigen3_POPULATED ${eigen3_POPULATED} CACHE BOOL "Whether eigen3 lib was manually loaded and installed" FORCE)

	execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${Eigen3_BASE_PATH}/build" "-S" "." "-DCMAKE_INSTALL_PREFIX=${Eigen3_BASE_PATH}/install" WORKING_DIRECTORY "${Eigen3_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${Eigen3_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${Eigen3_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

	if(DEBUG_INSTALLATION)
		message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
		message(STATUS "Build Error: ${BUILD_ERROR}")
		message(STATUS "Install Error: ${INSTALL_ERROR}")
	endif()

	set(Eigen3_DIR "${Eigen3_BASE_PATH}/build")
	find_package(Eigen3 3.4.0 QUIET REQUIRED PATHS ${Eigen3_DIR})

	unset(Eigen3_BASE_PATH)
endif()

PackageHelperTarget(Eigen3::Eigen EIGEN3_FOUND)

# # -----------------------------------------------------
# # -----------------------------------------------------

# BLAS and LAPACK (required for Windows)
if(WIN32)
	message(WARNING "The provided BLAS/LAPACK libraries for Windows only work for Intel CPUs. Please make sure to check the README.md file in external/thirdparty/win/BLAS.")
	set(blas_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/win/BLAS")
	set(lapack_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/win/LAPACK")

	if(${WINDOWS_ARCHITECTURE} STREQUAL "32Bit")
		LibraryHelper(BLAS_LIBRARIES BLAS.lib "${blas_BASE_PATH}/x32" blas_FOUND)
		LibraryHelper(LAPACK_LIBRARIES LAPACK.lib "${lapack_BASE_PATH}/x32" lapack_FOUND)
		LibraryHelper(LAPACKE_LIBRARIES LAPACKE.lib "${lapack_BASE_PATH}/x32" lapacke_FOUND)
	elseif(${WINDOWS_ARCHITECTURE} STREQUAL "64Bit")
		LibraryHelper(BLAS_LIBRARIES BLAS.lib "${blas_BASE_PATH}/x64" blas_FOUND)
		LibraryHelper(LAPACK_LIBRARIES LAPACK.lib "${lapack_BASE_PATH}/x64" lapack_FOUND)
		LibraryHelper(LAPACKE_LIBRARIES LAPACKE.lib "${lapack_BASE_PATH}/x64" lapacke_FOUND)
	endif()

	if(NOT blas_FOUND)
		message(FATAL_ERROR "Could not find BLAS. Please make sure to install an appropriate BLAS library for your CPU architecture.")
	endif()

	if(NOT lapack_FOUND)
		message(FATAL_ERROR "Could not find LAPACK. Please make sure to install an appropriate LAPACK library for your CPU architecture.")
	endif()

	if(NOT lapacke_FOUND)
		message(FATAL_ERROR "Could not find LAPACKE. Please make sure to install an appropriate LAPACKE library for your CPU architecture.")
	endif()

	PackageHelper(blas blas_FOUND "" "${BLAS_LIBRARIES}")
	PackageHelper(lapack lapack_FOUND "" "${LAPACK_LIBRARIES}")
	PackageHelper(lapacke lapacke_FOUND "" "${LAPACKE_LIBRARIES}")
endif()

# # -----------------------------------------------------
# # -----------------------------------------------------

# Matplotlib
set(MATPLOTLIB_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/matplotlib_cpp")
find_package(matplotlib_cpp QUIET HINTS "${MATPLOTLIB_BASE_PATH}/install/lib")

if((NOT matplotlib_cpp_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(matplotlib_cpp_FOUND)
	message("Manually download, configure and build Matplotlib library...")

	if(NOT matplotlib_POPULATED)
		FetchContent_Declare(
			matplotlib
			GIT_REPOSITORY "https://github.com/lava/matplotlib-cpp.git"
			GIT_TAG "ef0383f1315d32e0156335e10b82e90b334f6d9f"
			PREFIX "${MATPLOTLIB_BASE_PATH}"
			DOWNLOAD_DIR "${MATPLOTLIB_BASE_PATH}/download"
			SOURCE_DIR "${MATPLOTLIB_BASE_PATH}/source"
			BINARY_DIR "${MATPLOTLIB_BASE_PATH}/build"
			SUBBUILD_DIR "${MATPLOTLIB_BASE_PATH}/subbuild"
		)
		FetchContent_Populate(matplotlib)
	endif()

	execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${MATPLOTLIB_BASE_PATH}/build" "-S" "." "-DCMAKE_INSTALL_PREFIX=${MATPLOTLIB_BASE_PATH}/install" WORKING_DIRECTORY "${MATPLOTLIB_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${MATPLOTLIB_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${MATPLOTLIB_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

	if(DEBUG_INSTALLATION)
		message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
		message(STATUS "Build Error: ${BUILD_ERROR}")
		message(STATUS "Install Error: ${INSTALL_ERROR}")
	endif()

	set(matplotlib_POPULATED ${matplotlib_POPULATED} CACHE BOOL "Whether matplotlib was manually loaded and installed" FORCE)

	set(matplotlib_cpp_DIR "${MATPLOTLIB_BASE_PATH}/install/lib")
	find_package(matplotlib_cpp QUIET REQUIRED PATHS ${matplotlib_cpp_DIR})

	unset(MATPLOTLIB_BASE_PATH)
endif()

PackageHelperTarget(matplotlib_cpp::matplotlib_cpp matplotlib_cpp_FOUND)

# # -----------------------------------------------------
# # -----------------------------------------------------

# Opengl-lib
# add_subdirectory(${PROJECT_SOURCE_DIR}/external/opengl-lib/code)
# PackageHelperTarget(cppgl cppgl_FOUND)

# # -----------------------------------------------------
# # -----------------------------------------------------

# Nanoflann
set(nanoflann_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/include/nanoflann")

if(NOT EXISTS "${nanoflann_BASE_PATH}")
	message(FATAL_ERROR "The 'nanoflann' submodule was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

set(nanoflann_BASE_PATH ${PROJECT_SOURCE_DIR}/external/thirdparty/include/nanoflann)
option(NANOFLANN_BUILD_EXAMPLES "Build nanoflann examples" OFF)
add_subdirectory(${nanoflann_BASE_PATH})
PackageHelperTarget(nanoflann::nanoflann nanoflann_FOUND)
unset(NANOFLANN_BUILD_EXAMPLES)

# # -----------------------------------------------------
# # -----------------------------------------------------

# Json
set(json_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/include/json")

if(NOT EXISTS "${json_BASE_PATH}")
	message(FATAL_ERROR "The 'json' submodule was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

set(json_BASE_PATH ${PROJECT_SOURCE_DIR}/external/thirdparty/include/json)
IncludeHelper(json_INCLUDE_DIRS nlohmann/json.hpp "${json_BASE_PATH}/single_include" json_FOUND)
PackageHelper(json "${json_FOUND}" "${json_INCLUDE_DIRS}" "")

if(NOT json_FOUND)
	message(FATAL_ERROR "Could not find json.")
endif()

# # -----------------------------------------------------
# # -----------------------------------------------------

# # yaml parser
set(yaml_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/yaml-cpp")
find_package(yaml-cpp QUIET HINTS "${yaml_BASE_PATH}/install/share/cmake/yaml-cpp")

# download and install manually
if((NOT yaml-cpp_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(yaml-cpp_FOUND)

	if(NOT yaml_POPULATED)
		message("Manually download, configure and build yaml library...")
		FetchContent_Declare(
			yaml
			GIT_REPOSITORY "https://github.com/jbeder/yaml-cpp.git"
			GIT_TAG "yaml-cpp-0.7.0"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${yaml_BASE_PATH}/install"
			PREFIX "${yaml_BASE_PATH}"
			DOWNLOAD_DIR "${yaml_BASE_PATH}/download"
			SOURCE_DIR "${yaml_BASE_PATH}/source"
			BINARY_DIR "${yaml_BASE_PATH}/build"
			SUBBUILD_DIR "${yaml_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(yaml)
	endif()

	execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${yaml_BASE_PATH}/build" "-S" "."
		"-DCMAKE_INSTALL_PREFIX=${yaml_BASE_PATH}/install"
		"-DYAML_BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
		WORKING_DIRECTORY "${yaml_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${yaml_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${yaml_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

	if(DEBUG_INSTALLATION)
		message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
		message(STATUS "Build Error: ${BUILD_ERROR}")
		message(STATUS "Install Error: ${INSTALL_ERROR}")
	endif()

	set(yaml_POPULATED ${json_POPULATED} CACHE BOOL "Whether yaml lib was manually loaded and installed" FORCE)

	set(yaml_DIR "${yaml_BASE_PATH}/install/share/cmake/yaml-cpp")
	find_package(yaml-cpp QUIET REQUIRED PATHS ${yaml_DIR})

	unset(yaml_BASE_PATH)
endif()

PackageHelperTarget(yaml-cpp yaml-cpp_FOUND)

# # -----------------------------------------------------
# # -----------------------------------------------------

# # csv parser
set(rapidcsv_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/rapidcsv")
IncludeHelper(rapidcsv_INCLUDE_DIRS rapidcsv.h "${rapidcsv_BASE_PATH}/install/include" rapidcsv_FOUND)

# download and install manually
if((NOT rapidcsv_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(rapidcsv_FOUND)

	if(NOT rapidcsv_POPULATED)
		message("Manually download, configure and build rapidcsv library...")
		FetchContent_Declare(
			rapidcsv
			GIT_REPOSITORY "https://github.com/d99kris/rapidcsv.git"
			GIT_TAG "v8.75"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${rapidcsv_BASE_PATH}/install"
			PREFIX "${rapidcsv_BASE_PATH}"
			DOWNLOAD_DIR "${rapidcsv_BASE_PATH}/download"
			SOURCE_DIR "${rapidcsv_BASE_PATH}/source"
			BINARY_DIR "${rapidcsv_BASE_PATH}/build"
			SUBBUILD_DIR "${rapidcsv_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(rapidcsv)
	endif()

	execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${rapidcsv_BASE_PATH}/build" "-S" "."
		"-DCMAKE_INSTALL_PREFIX=${rapidcsv_BASE_PATH}/install"
		WORKING_DIRECTORY "${rapidcsv_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${rapidcsv_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${rapidcsv_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

	if(DEBUG_INSTALLATION)
		message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
		message(STATUS "Build Error: ${BUILD_ERROR}")
		message(STATUS "Install Error: ${INSTALL_ERROR}")
	endif()

	set(rapidcsv_POPULATED ${json_POPULATED} CACHE BOOL "Whether rapidcsv lib was manually loaded and installed" FORCE)

	set(rapidcsv_DIR "${rapidcsv_BASE_PATH}/install/")
	IncludeHelper(rapidcsv_INCLUDE_DIRS rapidcsv.h "${rapidcsv_BASE_PATH}/install/include" rapidcsv_FOUND)

	set(RAPIDCSV_INCLUDE_DIR ${rapidcsv_INCLUDE_DIRS})
	set(RAPIDCSV_INCLUDE_DIRS ${rapidcsv_INCLUDE_DIRS})

	if(NOT rapidcsv_FOUND)
		message(FATAL_ERROR "Could not find rapidcsv.")
	endif()

	unset(rapidcsv_BASE_PATH)
endif()

PackageHelper(rapidcsv "${rapidcsv_FOUND}" "${rapidcsv_INCLUDE_DIRS}" "")

# # -----------------------------------------------------
# # -----------------------------------------------------

# googletest
set(googletest_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/googletest")
find_package(GTest QUIET HINTS "${googletest_BASE_PATH}/install/lib/cmake/GTest")

# download and install manually
if((NOT GTest_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(GTest_FOUND)

	if(NOT googletest_POPULATED)
		message("Manually download, configure and build googletest library...")
		FetchContent_Declare(
			googletest
			GIT_REPOSITORY "https://github.com/google/googletest.git"
			GIT_TAG "release-1.11.0"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${googletest_BASE_PATH}/install"
			PREFIX "${googletest_BASE_PATH}"
			DOWNLOAD_DIR "${googletest_BASE_PATH}/download"
			SOURCE_DIR "${googletest_BASE_PATH}/source"
			BINARY_DIR "${googletest_BASE_PATH}/build"
			SUBBUILD_DIR "${googletest_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(googletest)
	endif()

	execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${googletest_BASE_PATH}/build" "-S" "." "-DCMAKE_INSTALL_PREFIX=${googletest_BASE_PATH}/install" WORKING_DIRECTORY "${googletest_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${googletest_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${googletest_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

	if(DEBUG_INSTALLATION)
		message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
		message(STATUS "Build Error: ${BUILD_ERROR}")
		message(STATUS "Install Error: ${INSTALL_ERROR}")
	endif()

	set(googletest_POPULATED ${googletest_POPULATED} CACHE BOOL "Whether googletest lib was manually loaded and installed" FORCE)

	set(googletest_DIR "${googletest_BASE_PATH}/install/lib/cmake/GTest")
	find_package(GTest QUIET REQUIRED PATHS ${googletest_DIR})

	unset(googletest_BASE_PATH)
endif()

PackageHelperTarget(GTest::gtest GTest_FOUND)

# # -----------------------------------------------------
# # -----------------------------------------------------

# Sophus
set(sophus_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/sophus")
find_package(Sophus QUIET HINTS "${sophus_BASE_PATH}/install/lib/cmake/Sophus")

# download and install manually
if((NOT Sophus_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(Sophus_FOUND)

	if(NOT sophus_POPULATED)
		message("Manually download, configure and build Sophus library...")
		FetchContent_Declare(
			sophus
			GIT_REPOSITORY "https://github.com/strasdat/Sophus.git"
			GIT_TAG "v1.0.0"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${sophus_BASE_PATH}/install"
			PREFIX "${sophus_BASE_PATH}"
			DOWNLOAD_DIR "${sophus_BASE_PATH}/download"
			SOURCE_DIR "${sophus_BASE_PATH}/source"
			BINARY_DIR "${sophus_BASE_PATH}/build"
			SUBBUILD_DIR "${sophus_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(sophus)
	endif()

	get_target_property(EIGEN3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
	execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${sophus_BASE_PATH}/build" "-S" "." "-DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIRS}" "-DCMAKE_INSTALL_PREFIX=${sophus_BASE_PATH}/install" WORKING_DIRECTORY "${sophus_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${sophus_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${sophus_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

	if(DEBUG_INSTALLATION)
		message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
		message(STATUS "Build Error: ${BUILD_ERROR}")
		message(STATUS "Install Error: ${INSTALL_ERROR}")
	endif()

	set(sophus_POPULATED ${sophus_POPULATED} CACHE BOOL "Whether Sophus lib was manually loaded and installed" FORCE)

	set(sophus_DIR "${sophus_BASE_PATH}/install/lib/cmake/Sophus")
	find_package(Sophus QUIET REQUIRED PATHS ${sophus_DIR})

	unset(sophus_BASE_PATH)
endif()

PackageHelper(Sophus "${Sophus_FOUND}" "${Sophus_INCLUDE_DIRS}" "")

# # -----------------------------------------------------
# # -----------------------------------------------------

# zlib
set(zlib_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/zlib")

if(UNIX)
	set(zlib_libs libz.so)
	LibraryHelper(zlib_LIBRARIES zlib_libs "${zlib_BASE_PATH}/install/lib" zlib_FOUND)
elseif(WIN32)
	LibraryHelper(zlib_LIBRARIES zlib.lib "${zlib_BASE_PATH}/install/lib" zlib_FOUND)
	set(ZLIB_LIBRARY ${zlib_LIBRARIES})
endif()

IncludeHelper(zlib_INCLUDE_DIRS zlib.h "${zlib_BASE_PATH}/install/include" zlib_FOUND)
set(zlib_DIR "${zlib_BASE_PATH}/install")
set(ZLIB_DIR "${zlib_BASE_PATH}/install")

# download and install manually
if((NOT zlib_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIE) OR FORCE_MANUAL_INSTALLATION)
	unset(zlib_FOUND)

	if(NOT zlib_POPULATED)
		message("Manually download, configure and build zlib library...")
		FetchContent_Declare(
			zlib
			GIT_REPOSITORY "https://github.com/madler/zlib.git"
			GIT_TAG "v1.2.12"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${zlib_BASE_PATH}/install"
			PREFIX "${zlib_BASE_PATH}"
			DOWNLOAD_DIR "${zlib_BASE_PATH}/download"
			SOURCE_DIR "${zlib_BASE_PATH}/source"
			BINARY_DIR "${zlib_BASE_PATH}/build"
			SUBBUILD_DIR "${zlib_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(zlib)
	endif()

	execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${zlib_BASE_PATH}/build" "-S" "." "-DCMAKE_INSTALL_PREFIX=${zlib_BASE_PATH}/install" WORKING_DIRECTORY "${zlib_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${zlib_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${zlib_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

	if(DEBUG_INSTALLATION)
		message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
		message(STATUS "Build Error: ${BUILD_ERROR}")
		message(STATUS "Install Error: ${INSTALL_ERROR}")
	endif()

	set(zlib_POPULATED ${zlib_POPULATED} CACHE BOOL "Whether zlib lib was manually loaded and installed" FORCE)

	set(zlib_DIR "${zlib_BASE_PATH}/install")
	set(ZLIB_DIR "${zlib_BASE_PATH}/install")

	if(UNIX)
		set(zlib_libs libz.so)
		LibraryHelper(zlib_LIBRARIES zlib_libs "${zlib_DIR}/lib" zlib_FOUND)
	elseif(WIN32)
		LibraryHelper(zlib_LIBRARIES zlib.lib "${zlib_DIR}/lib" zlib_FOUND)
		set(ZLIB_LIBRARY ${zlib_LIBRARIES})
	endif()

	IncludeHelper(zlib_INCLUDE_DIRS zlib.h "${zlib_DIR}/include" zlib_FOUND)
	set(ZLIB_INCLUDE_DIR ${zlib_INCLUDE_DIRS})

	if(NOT zlib_FOUND)
		message(FATAL_ERROR "Could not find zlib.")
	endif()

	unset(zlib_BASE_PATH)
endif()

PackageHelper(zlib "${zlib_FOUND}" "${zlib_INCLUDE_DIRS}" "${zlib_LIBRARIES}")
unset(zlib_libs)

# # -----------------------------------------------------
# # -----------------------------------------------------

# cnpy
if(UNIX)
	set(cnpy_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/cnpy")

	if(UNIX)
		LibraryHelper(cnpy_LIBRARIES libcnpy.so "${cnpy_BASE_PATH}/build/lib" cnpy_FOUND)
	elseif(WIN32)
		LibraryHelper(cnpy_LIBRARIES cnpy.lib "${cnpy_BASE_PATH}/build/lib" cnpy_FOUND)

		# FileHelper(cnpy_LIBRARIES cnpy.dll "${cnpy_BASE_PATH}/build/bin" cnpy_FOUND)
		file(COPY "${cnpy_BASE_PATH}/build/bin/cnpy.dll" DESTINATION ${OUTPUT_DIR})
	endif()

	IncludeHelper(cnpy_INCLUDE_DIRS cnpy.h "${cnpy_BASE_PATH}/build/include" cnpy_FOUND)

	if(cnpy_FOUND)
		set(cnpy_DIR "${cnpy_BASE_PATH}/build")
	endif()

	# download and install manually
	if((NOT cnpy_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
		unset(cnpy_FOUND)

		if(NOT cnpy_POPULATED)
			message("Manually download, configure and build cnpy library...")
			FetchContent_Declare(
				cnpy
				GIT_REPOSITORY "https://github.com/rogersce/cnpy.git"
				GIT_TAG "4e8810b1a8637695171ed346ce68f6984e585ef4"
				CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${cnpy_BASE_PATH}/build"
				PREFIX "${cnpy_BASE_PATH}"
				DOWNLOAD_DIR "${cnpy_BASE_PATH}/download"
				SOURCE_DIR "${cnpy_BASE_PATH}/source"
				BINARY_DIR "${cnpy_BASE_PATH}/build"
				SUBBUILD_DIR "${cnpy_BASE_PATH}/subbuild"
			)

			FetchContent_Populate(cnpy)
		endif()

		get_target_property(EIGEN3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
		execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${cnpy_BASE_PATH}/build" "-S" "."
			"-DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIRS}" "-DZLIB_LIBRARY=${zlib_LIBRARIES}" "-DZLIB_INCLUDE_DIR=${zlib_INCLUDE_DIRS}"
			"-DCMAKE_INSTALL_PREFIX=${cnpy_BASE_PATH}/build" WORKING_DIRECTORY "${cnpy_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
		execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${cnpy_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
		execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${cnpy_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

		if(DEBUG_INSTALLATION)
			message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
			message(STATUS "Build Error: ${BUILD_ERROR}")
			message(STATUS "Install Error: ${INSTALL_ERROR}")
		endif()

		set(cnpy_POPULATED ${cnpy_POPULATED} CACHE BOOL "Whether cnpy lib was manually loaded and installed" FORCE)

		set(cnpy_DIR "${cnpy_BASE_PATH}/build")

		if(UNIX)
			set(cnpy_libs libcnpy.so)
			LibraryHelper(cnpy_LIBRARIES cnpy_libs "${cnpy_DIR}/lib" cnpy_FOUND)
		elseif(WIN32)
			LibraryHelper(cnpy_LIBRARIES cnpy.lib "${cnpy_BASE_PATH}/build/lib" cnpy_FOUND)

			# FileHelper(cnpy_LIBRARIES cnpy.dll "${cnpy_BASE_PATH}/build/bin" cnpy_FOUND)
			file(COPY "${cnpy_BASE_PATH}/build/bin/cnpy.dll" DESTINATION ${OUTPUT_DIR})
		endif()

		IncludeHelper(cnpy_INCLUDE_DIRS cnpy.h "${cnpy_BASE_PATH}/build/include" cnpy_FOUND)

		if(NOT cnpy_FOUND)
			message(FATAL_ERROR "Could not find cnpy.")
		endif()

		unset(cnpy_BASE_PATH)
	endif()

	unset(cnpy_libs)
	PackageHelper(cnpy "${cnpy_FOUND}" "${cnpy_INCLUDE_DIRS}" "${cnpy_LIBRARIES}")

elseif(WIN32)
	message(WARNING "Warning: Cnpy is not supported for Windows yet.")
endif()

unset(DEBUG_INSTALLATION)
unset(BUILD_SHARED_LIBS)

set(CORE_INCLUDES ${PACKAGE_INCLUDES})
set(CORE_LIBS ${LIBS})
set(CORE_TARGETS ${LIB_TARGETS})
set(MODULE_CORE 1)

# message(STATUS "${LIBS}")
