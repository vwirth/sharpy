# Defines the following output variables:
#
# OPENCV_INCLUES:    The list of required include directories
# OPENCV_LIBS:        The list of required libraries for link_target
# OPENCV_TARGETS:     The list of required targets
# MODULE_OPENCV:      True if all required dependencies are found.
#

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_OPENCV)

set(DEBUG_INSTALLATION ON)
set(FORCE_MANUAL_INSTALLATION OFF)
set(BUILD_SHARED_LIBS ON)

include(FetchContent)

# allow no updates -> faster
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# OpenCV
set(OpenCV_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/opencv")

if(UNIX)
    find_package(OpenCV 4.5 QUIET HINTS "${OpenCV_BASE_PATH}/install/lib/cmake/opencv4")
elseif(WIN32)
    find_package(OpenCV 4.5 QUIET HINTS "${OpenCV_BASE_PATH}/build")
endif()

if((NOT OPENCV_FOUND AND INSTALL_MISSING_REQUIRED_DEPENDENCIES) OR FORCE_MANUAL_INSTALLATION)
    unset(OPENCV_FOUND)
    set(OpenCV_CONTRIB_BASE_PATH "${PROJECT_SOURCE_DIR}/external/thirdparty/opencv_contrib")

    # set(CMAKE_EXECUTE_PROCESS_COMMAND_ECHO STDOUT)
    message("Manually download, configure and build OpenCV library...")

    if(NOT opencvcontrib_POPULATED)
        FetchContent_Declare(
            OpenCVContrib
            GIT_REPOSITORY "https://github.com/opencv/opencv_contrib"
            GIT_TAG "4.5.0"
            PREFIX "${OpenCV_CONTRIB_BASE_PATH}"
            DOWNLOAD_DIR "${OpenCV_CONTRIB_BASE_PATH}/download"
            SOURCE_DIR "${OpenCV_CONTRIB_BASE_PATH}/source"
            BINARY_DIR "${OpenCV_CONTRIB_BASE_PATH}/build"
            SUBBUILD_DIR "${OpenCV_CONTRIB_BASE_PATH}/subbuild"
            INSTALL_COMMAND "${CMAKE_COMMAND} --target install"
        )
        FetchContent_Populate(OpenCVContrib)
    endif()

    set(opencvcontrib_POPULATED ${opencvcontrib_POPULATED} CACHE BOOL "Whether OpenCV contrib lib was manually loaded and installed" FORCE)

    if(NOT opencv_POPULATED)
        FetchContent_Declare(
            OpenCV
            GIT_REPOSITORY "https://github.com/opencv/opencv"
            GIT_TAG "4.5.0"
            PREFIX "${OpenCV_BASE_PATH}"
            DOWNLOAD_DIR "${OpenCV_BASE_PATH}/download"
            SOURCE_DIR "${OpenCV_BASE_PATH}/source"
            BINARY_DIR "${OpenCV_BASE_PATH}/build"
            SUBBUILD_DIR "${OpenCV_BASE_PATH}/subbuild"
            INSTALL_COMMAND "${CMAKE_COMMAND} --target install"
        )
        FetchContent_Populate(OpenCV)
    endif()

    message(STATUS "CUDA GENERATION: ${CUDA_ARCHITECTURE}")
    message(STATUS "Installing OpenCV with args: ${CMAKE_ARGS}")
    execute_process(COMMAND "${CMAKE_COMMAND}" "-B" "${OpenCV_BASE_PATH}/build" "-S" "."
        "-DBUILD_SHARED_LIBS=ON"
        "-DBUILD_PERF_TESTS=OFF" "-DBUILD_TESTS=OFF"
        "-DCMAKE_INSTALL_PREFIX=${OpenCV_BASE_PATH}/install"
        "-DOPENCV_EXTRA_MODULES_PATH=${OpenCV_CONTRIB_BASE_PATH}/source/modules"
        "-DWITH_EIGEN=ON" "-DWITH_CUDA=ON" "-DWITH_CUBLAS=ON" "-DWITH_CUFFT=ON"
        "-DWITH_CUDNN=OFF" "-DENABLE_FAST_MATH=ON" "-DCUDA_FAST_MATH=ON"
        "-DBLAS_LIBRARIES=${BLAS_LIBRARIES}" "-DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}" "-DBUILD_JAVA=OFF"
        "-DWITH_VTK=OFF" "-DWITH_PYTHON=OFF" "-DBUILD_opencv_python2=OFF" "-DBUILD_opencv_python3=OFF"
        "-DCUDA_GENERATION=${CUDA_ARCHITECTURE}" "-DINSTALL_PYTHON_EXAMPLES=OFF"
        "-DCUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE=OFF"
        "-DOPENCV_CUDA_ARCH_BIN=${CMAKE_CUDA_ARCHITECTURES}"
        "-DCUDA_ARCH_BIN=${CMAKE_CUDA_ARCHITECTURES}"
        "-DCUDA_ARCH_PTX=\"\""
        "-DBUILD_opencv_python_bindings_generator=OFF" "-DBUILD_opencv_python_tests=OFF"
        "-DCMAKE_USE_WIN32_THREADS_INIT=${WIN32}"
        "-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}"
        WORKING_DIRECTORY "${OpenCV_BASE_PATH}/source" ERROR_VARIABLE CONFIGURE_ERROR)
    unset(CMAKE_ARGS)
    execute_process(COMMAND "${CMAKE_COMMAND}" "--build" "." "--config" ${VSMODE} "-j" "8" WORKING_DIRECTORY "${OpenCV_BASE_PATH}/build" ERROR_VARIABLE BUILD_ERROR)
    execute_process(COMMAND "${CMAKE_COMMAND}" "--install" "." WORKING_DIRECTORY "${OpenCV_BASE_PATH}/build" ERROR_VARIABLE INSTALL_ERROR)

    if(DEBUG_INSTALLATION)
        message(STATUS "Configure Error: ${CONFIGURE_ERROR}")
        message(STATUS "Build Error: ${BUILD_ERROR}")
        message(STATUS "Install Error: ${INSTALL_ERROR}")
    endif()

    set(opencv_POPULATED ${opencv_POPULATED} CACHE BOOL "Whether OpenCV lib was manually loaded and installed" FORCE)

    if(UNIX)
        set(OpenCV_DIR "${OpenCV_BASE_PATH}/install/lib/cmake/opencv4")
    elseif(WIN32)
        set(OpenCV_DIR "${OpenCV_BASE_PATH}/build")
    endif()

    find_package(OpenCV 4.5.0 QUIET REQUIRED PATHS ${OpenCV_DIR})

    unset(OpenCV_BASE_PATH)
endif()

# opencv has a bunch of unnecessary libs that we do not want to link against to since it's taking too much time
set(PROJECT_LIBS
    "opencv_core"
    "opencv_imgproc"
    "opencv_cudaarithm"
    "opencv_cudaimgproc"
    "opencv_calib3d"
    "opencv_objdetect"
    "opencv_video"
    "opencv_cudev"
    "opencv_highgui"
    "opencv_imgcodecs"
    "opencv_cudawarping"
    "opencv_features2d"
    "opencv_dnn"
    "opencv_videoio"
)

foreach(lib IN LISTS OpenCV_LIBS)
    # foreach(accepted_lib IN LISTS PROJECT_LIBS)
    # if("${lib}" STREQUAL "${accepted_lib}")
    # # message(STATUS "here is a lib: ${lib}")
    # PackageHelperTarget(${lib} OPENCV_FOUND)
    # break()
    # else()
    # # message(STATUS "REJECTED ${lib}")
    # endif()
    # endforeach()
    PackageHelperTarget(${lib} OPENCV_FOUND)
endforeach()

# opencv has a bunch of unnecessary libs that we do not need to link
# the following lines instead of the previous three decreasea the number of linked opencv libs from 65 to 12
# However, no link-time improvement can be measured
# set(PROJECT_LIBS
# "opencv_core"
# "opencv_imgproc"
# "opencv_cudaarithm"
# "opencv_cudaimgproc"
#
# # "opencv_video"
# "opencv_cudev"
# "opencv_highgui"
# "opencv_imgcodecs"
# "opencv_cudawarping"
# "opencv_features2d"
#
# # "opencv_videoio"
# )
#
# foreach(lib IN LISTS OpenCV_LIBS)
# foreach(accepted_lib IN LISTS PROJECT_LIBS)
# if("${lib}" STREQUAL "${accepted_lib}")
# message(STATUS "here is a lib: ${lib}")
# PackageHelperTarget(${lib} OPENCV_FOUND)
# break()
# else()
# message(STATUS "REJECTED ${lib}")
# endif()
# endforeach()
# endforeach()
unset(DEBUG_INSTALLATION)
unset(BUILD_SHARED_LIBS)

set(OPENCV_INCLUDES ${PACKAGE_INCLUDES})
set(OPENCV_LIBS ${LIBS})
set(OPENCV_TARGETS ${LIB_TARGETS})
set(MODULE_OPENCV 1)