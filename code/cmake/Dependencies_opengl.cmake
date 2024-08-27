# Defines the following output variables:
#
# OPENGL_INCLUDES:    The list of required include directories
# OPENGL_LIBS:        The list of required libraries for link_target
# OPENGL_TARGETS:     The list of required targets
# MODULE_OPENGL:      True if all required dependencies are found.
#

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_OPENGL)

set(DEBUG_INSTALLATION ON)
set(FORCE_MANUAL_INSTALLATION OFF)
set(BUILD_SHARED_LIBS ON)

include(FetchContent)

# allow no updates -> faster
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# # -----------------------------------------------------
# # -----------------------------------------------------

# # OpenGL
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
if(UNIX)
	LibraryHelper(OPENGL_LIBRARIES libEGL.so "" OPENGL_FOUND)
endif()
PackageHelper(OpenGL OPENGL_FOUND "${OPENGL_INCLUDE_DIR}" "${OPENGL_LIBRARIES}")


# # -----------------------------------------------------
# # -----------------------------------------------------

# # FreeImage
if(UNIX)
	IncludeHelper(freeimage_INCLUDE_DIRS FreeImage.h "" freeimage_FOUND)
	LibraryHelper(freeimage_LIBRARIES libfreeimage.so "" freeimage_FOUND)
	LibraryHelper(freeimage_LIBRARIES libtiff.so "" freeimage_FOUND)
elseif(WIN32)
	set(freeimage_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/win/FreeImage/Dist")

	if(${WINDOWS_ARCHITECTURE} STREQUAL "32Bit")
		IncludeHelper(freeimage_INCLUDE_DIRS FreeImage.h "${freeimage_BASE_PATH}/x32" freeimage_FOUND)

		# FileHelper(freeimage_LIBRARIES FreeImage.dll "${freeimage_BASE_PATH}/x32" freeimage_FOUND)
		file(COPY "${freeimage_BASE_PATH}/x32/FreeImage.dll" DESTINATION ${OUTPUT_DIR})
		LibraryHelper(freeimage_LIBRARIES FreeImage.lib "${freeimage_BASE_PATH}/x32" freeimage_FOUND)
	elseif(${WINDOWS_ARCHITECTURE} STREQUAL "64Bit")
		IncludeHelper(freeimage_INCLUDE_DIRS FreeImage.h "${freeimage_BASE_PATH}/x64" freeimage_FOUND)

		# FileHelper(freeimage_LIBRARIES FreeImage.dll "${freeimage_BASE_PATH}/x64" freeimage_FOUND)
		file(COPY "${freeimage_BASE_PATH}/x64/FreeImage.dll" DESTINATION ${OUTPUT_DIR})
		LibraryHelper(freeimage_LIBRARIES FreeImage.lib "${freeimage_BASE_PATH}/x64" freeimage_FOUND)
	endif()
endif()

if(NOT freeimage_FOUND)
	message(FATAL_ERROR "Could not find FreeImage.")
endif()

PackageHelper(FreeImage freeimage_FOUND "${freeimage_INCLUDE_DIRS}" "${freeimage_LIBRARIES}")

# # -----------------------------------------------------
# # -----------------------------------------------------

# # Imgui (should be already included in header-only libraries of dependencies_code)
set(imgui_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/include/imgui")

if(NOT EXISTS "${imgui_BASE_PATH}")
	message(FATAL_ERROR "The 'imgui' submodule was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

PackageInfo(imgui true "${imgui_BASE_PATH}" "")
IncludeHelper(imgui_INCLUDE_DIRS imgui.h ${imgui_BASE_PATH} imgui_FOUND)
PackageHelper(imgui_backend_headers imgui_FOUND "${imgui_INCLUDE_DIRS}" "")

if(NOT imgui_FOUND)
	message(FATAL_ERROR "Could not find imgui.")
endif()

# # stb (should be already included in header-only libraries of dependencies_code)
set(stb_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/include/stb")

if(NOT EXISTS "${stb_BASE_PATH}")
	message(FATAL_ERROR "The 'stb' submodule was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

PackageInfo(stb true "${stb_BASE_PATH}" "")
IncludeHelper(stb_INCLUDE_DIRS stb_image_write.h ${stb_BASE_PATH} stb_FOUND)

if(NOT stb_FOUND)
	message(FATAL_ERROR "Could not find stb.")
endif()

# glfw
set(glfw3_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/glfw")
find_package(glfw3 QUIET HINTS "${glfw3_BASE_PATH}/install/lib/cmake/glfw3")

# download and install manually
if((NOT glfw3_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(glfw3_FOUND)

	if(NOT glfw3_POPULATED)
		message("Manually download, configure and build glfw3 library...")
		FetchContent_Declare(
			glfw3
			GIT_REPOSITORY "https://github.com/glfw/glfw.git"
			GIT_TAG "4cb36872a5fe448c205d0b46f0e8c8b57530cfe0"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${glfw3_BASE_PATH}/install"
			PREFIX "${glfw3_BASE_PATH}"
			DOWNLOAD_DIR "${glfw3_BASE_PATH}/download"
			SOURCE_DIR "${glfw3_BASE_PATH}/source"
			BINARY_DIR "${glfw3_BASE_PATH}/build"
			SUBBUILD_DIR "${glfw3_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(glfw3)
	endif()

	execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${glfw3_BASE_PATH}/build" "-S" "." "-DCMAKE_INSTALL_PREFIX=${glfw3_BASE_PATH}/install" WORKING_DIRECTORY "${glfw3_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${glfw3_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${glfw3_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

	if(DEBUG_INSTALLATION)
		message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
		message(STATUS "Build Error: ${BUILD_ERROR}")
		message(STATUS "Install Error: ${INSTALL_ERROR}")
	endif()

	set(glfw3_POPULATED ${glfw_POPULATED} CACHE BOOL "Whether glfw3 lib was manually loaded and installed" FORCE)

	set(glfw3_DIR "${glfw3_BASE_PATH}/install/lib/cmake/glfw3")
	find_package(glfw3 QUIET REQUIRED PATHS ${glfw3_DIR})

	unset(glfw3_BASE_PATH)
endif()

PackageHelperTarget(glfw glfw_FOUND)

# glew
set(glew_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/glew")
find_package(glew QUIET HINTS "${glew_BASE_PATH}/install/lib/cmake/glew")

# download and install manually
if((NOT glew_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(glew_FOUND)

	if(NOT glew_POPULATED)
		message("Manually download, configure and build glew library...")
		FetchContent_Declare(
			glew
			URL "https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.zip"
			GIT_TAG "2c4c183c342b8ccf32372143ea7b365c3d5c7c41"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${glew_BASE_PATH}/install"
			PREFIX "${glew_BASE_PATH}"
			DOWNLOAD_DIR "${glew_BASE_PATH}/download"
			SOURCE_DIR "${glew_BASE_PATH}/source"
			BINARY_DIR "${glew_BASE_PATH}/build"
			SUBBUILD_DIR "${glew_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(glew)
	endif()

	execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${glew_BASE_PATH}/build" "-S" "." "-DCMAKE_INSTALL_PREFIX=${glew_BASE_PATH}/install" "-DOpenGL_GL_PREFERENCE='GLVND'" WORKING_DIRECTORY "${glew_BASE_PATH}/source/build/cmake" ERROR_VARIABLE CONFIGURE_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${glew_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${glew_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

	if(DEBUG_INSTALLATION)
		message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
		message(STATUS "Build Error: ${BUILD_ERROR}")
		message(STATUS "Install Error: ${INSTALL_ERROR}")
	endif()

	set(glew_POPULATED ${glew_POPULATED} CACHE BOOL "Whether glew lib was manually loaded and installed" FORCE)

	set(glew_DIR "${glew_BASE_PATH}/install/lib/cmake/glew")
	find_package(glew QUIET REQUIRED PATHS ${glew_DIR})

	unset(glew_BASE_PATH)
endif()

PackageHelperTarget(GLEW::glew glew_FOUND)

# # -----------------------------------------------------
# # -----------------------------------------------------

# assimp
set(ENV{ZLIB_HOME} "${zlib_DIR}")
set(assimp_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/assimp")
find_package(assimp QUIET PATHS "${assimp_BASE_PATH}/build/lib/cmake/assimp-5.2" NO_DEFAULT_PATH)

# download and install manually
if((NOT assimp_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
	unset(assimp_FOUND)

	if(NOT assimp_POPULATED)
		message("Manually download, configure and build assimp library...")
		FetchContent_Declare(
			assimp
			GIT_REPOSITORY "https://github.com/assimp/assimp.git"
			GIT_TAG "75ab2beb062d56abfe717355ccfb574980f8f2fd"
			CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${assimp_BASE_PATH}/install"
			PREFIX "${assimp_BASE_PATH}"
			DOWNLOAD_DIR "${assimp_BASE_PATH}/download"
			SOURCE_DIR "${assimp_BASE_PATH}/source"
			BINARY_DIR "${assimp_BASE_PATH}/build"
			SUBBUILD_DIR "${assimp_BASE_PATH}/subbuild"
		)

		FetchContent_Populate(assimp)
	endif()

	execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${assimp_BASE_PATH}/build" "-S" "."
		"-DCMAKE_INSTALL_PREFIX=${assimp_BASE_PATH}/build"
		"-DCMAKE_CXX_FLAGS_RELEASE=\"/MT\""
		"-DASSIMP_BUILD_ZLIB=ON"
		WORKING_DIRECTORY "${assimp_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${assimp_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
	execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${assimp_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

	if(DEBUG_INSTALLATION)
		message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
		message(STATUS "Build Error: ${BUILD_ERROR}")
		message(STATUS "Install Error: ${INSTALL_ERROR}")
	endif()

	set(assimp_POPULATED ${assimp_POPULATED} CACHE BOOL "Whether assimp lib was manually loaded and installed" FORCE)

	set(assimp_DIR "${assimp_BASE_PATH}/build/lib/cmake/assimp-5.2")
	find_package(assimp QUIET REQUIRED PATHS ${assimp_DIR} NO_DEFAULT_PATH)

	unset(assimp_BASE_PATH)
endif()

PackageHelperTarget(assimp::assimp assimp_FOUND)

unset(DEBUG_INSTALLATION)
unset(BUILD_SHARED_LIBS)

set(OPENGL_INCLUDES ${PACKAGE_INCLUDES})
set(OPENGL_LIBS ${LIBS})
set(OPENGL_TARGETS ${LIB_TARGETS})
set(MODULE_OPENGL 1)
