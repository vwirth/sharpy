# Defines the following output variables:
#
# CAMERA_INCLUDES:    The list of required include directories
# CAMERA_LIBS:        The list of required libraries for link_target
# CAMERA_TARGETS:     The list of required targets
# MODULE_CAMERA:      True if all required dependencies are found.
#

unset(PACKAGE_INCLUDES)
unset(LIB_TARGETS)
unset(LIBS)
unset(MODULE_CAMERA)

set(DEBUG_INSTALLATION ON)
set(FORCE_MANUAL_INSTALLATION OFF)
set(BUILD_SHARED_LIBS ON)

include(FetchContent)

# allow no updates -> faster
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# -----------------------------------------------------
# # -----------------------------------------------------

# Freenect2
find_package(freenect2)
PackageHelper(freenect2 "${freenect2_FOUND}" "${freenect2_INCLUDE_DIRS}" "${freenect2_LIBRARIES}")

# -----------------------------------------------------
# # -----------------------------------------------------

# Kinect Azure Development Kit
find_package(k4a)
PackageHelperTarget(k4a::k4a k4a_FOUND)

# -----------------------------------------------------
# # -----------------------------------------------------
unset(DEBUG_INSTALLATION)
unset(BUILD_SHARED_LIBS)

set(CAMERA_INCLUDES ${PACKAGE_INCLUDES})
set(CAMERA_LIBS ${LIBS})
set(CAMERA_TARGETS ${LIB_TARGETS})
set(MODULE_CAMERA 1)

# message(STATUS "${LIBS}")
