macro(GroupSources startDir curdir)
    file(GLOB children RELATIVE ${startDir}/${curdir} ${startDir}/${curdir}/*)

    foreach(child ${children})
        if(IS_DIRECTORY ${startDir}/${curdir}/${child})
            GroupSources(${startDir} ${curdir}/${child})
        else()
            string(REPLACE "/" "\\" groupname ${curdir})
            source_group(${groupname} FILES ${startDir}/${curdir}/${child})
        endif()
    endforeach()
endmacro()

macro(GroupSources2 startDir)
    file(GLOB children RELATIVE ${startDir} ${startDir}/*)

    foreach(child ${children})
        if(IS_DIRECTORY ${startDir}/${child})
            GroupSources(${startDir} ${child})
        else()
            source_group("" FILES ${startDir}/${child})
        endif()
    endforeach()
endmacro()

macro(OptionsHelper _variableName _description _defaultValue)
    option(${_variableName} "${_description}" "${_defaultValue}")

    # Create a padding string to align the console output
    string(LENGTH ${_variableName} SIZE)
    math(EXPR SIZE 25-${SIZE})
    string(RANDOM LENGTH ${SIZE} ALPHABET " " padding)
    message(STATUS "Option ${_variableName} ${padding} ${${_variableName}}")
endmacro()

macro(LibraryHelper _var _names _hints _found)
    set(tmp_var "${_var}-NOTFOUND")
    list(LENGTH ${_names} list_length)
    list(LENGTH ${_var} var_length)

    # check whether names is a list or just a single name
    if(list_length EQUAL 0)
        find_library(tmp_var NAMES ${_names} HINTS ${_hints})
    else()
        find_library(tmp_var NAMES ${${_names}} HINTS ${_hints})
    endif()

    unset(list_length)

    if(${tmp_var} STREQUAL "tmp_var-NOTFOUND") # if(${tmp_var} STREQUAL "${_var}-NOTFOUND")
        set(${_found} FALSE)
    elseif(NOT DEFINED ${_found})
        set(${_found} TRUE)
    endif()

    if(NOT(${tmp_var} STREQUAL "tmp_var-NOTFOUND"))
        if(var_length EQUAL 0)
            set(${_var} ${tmp_var})
        else()
            set(${_var} ${${_var}} ${tmp_var})
        endif()
    endif()

    unset(var_length)
    set(tmp_var "${_var}-NOTFOUND")

    # if(${_found} AND ${ARGC} GREATER 4)
    # get_filename_component(tmp ${${_var}} DIRECTORY)
    # get_filename_component(tmp_name ${tmp} NAME)

    # # if(${tmp_name} STREQUAL "lib")
    # # get_filename_component(tmp ${tmp} DIRECTORY)
    # # endif()
    # set(${ARGV4} ${tmp})
    # unset(tmp)
    # unset(tmp_name)
    # endif()
endmacro()

macro(FileHelper _var _names _hints _found)
    set(tmp_var "${_var}-NOTFOUND")
    list(LENGTH ${_names} list_length)
    list(LENGTH ${_var} var_length)

    # check whether names is a list or just a single name
    if(list_length EQUAL 0)
        find_file(tmp_var NAMES ${_names} HINTS ${_hints})
    else()
        find_file(tmp_var NAMES ${${_names}} HINTS ${_hints})
    endif()

    unset(list_length)

    if(${tmp_var} STREQUAL "tmp_var-NOTFOUND") # if(${tmp_var} STREQUAL "${_var}-NOTFOUND")
        set(${_found} FALSE)
    elseif(NOT DEFINED ${_found})
        set(${_found} TRUE)
    endif()

    if(NOT(${tmp_var} STREQUAL "tmp_var-NOTFOUND"))
        if(var_length EQUAL 0)
            set(${_var} ${tmp_var})
        else()
            set(${_var} ${${_var}} ${tmp_var})
        endif()
    endif()

    unset(var_length)
    set(tmp_var "${_var}-NOTFOUND")
    unset(tmp_var)

    # if(${_found} AND ${ARGC} GREATER 4)
    # get_filename_component(tmp ${${_var}} DIRECTORY)
    # get_filename_component(tmp_name ${tmp} NAME)

    # # if(${tmp_name} STREQUAL "lib")
    # # get_filename_component(tmp ${tmp} DIRECTORY)
    # # endif()
    # set(${ARGV4} ${tmp})
    # unset(tmp)
    # unset(tmp_name)
    # endif()
endmacro()

# optional 5th argument: _dir
# similar to find_package, this macro sets the
# <package-name>_DIR variable as 5th argument
# since this might not work every time since every
# package path is individual, the argument is set to optional
macro(IncludeHelper _var _names _hints _found)
    set(tmp_var "${_var}-NOTFOUND")
    list(LENGTH ${_names} list_length)
    list(LENGTH ${_var} var_length)

    # check whether names is a list or just a single name
    if(list_length EQUAL 0)
        find_path(tmp_var NAMES ${_names} HINTS ${_hints})
    else()
        find_path(tmp_var NAMES ${${_names}} HINTS ${_hints})
    endif()

    unset(list_length)

    if(${tmp_var} STREQUAL "tmp_var-NOTFOUND") # if(${tmp_var} STREQUAL "${_var}-NOTFOUND")
        set(${_found} FALSE)
    elseif(NOT DEFINED ${_found})
        set(${_found} TRUE)
    endif()

    if(NOT(${tmp_var} STREQUAL "tmp_var-NOTFOUND"))
        if(var_length EQUAL 0)
            set(${_var} "${tmp_var}")
        else()
            set(${_var} ${${_var}} "${tmp_var}")
        endif()
    endif()

    unset(var_length)
    set(tmp_var "${_var}-NOTFOUND")
    unset(tmp_var)

    # if(${_found} AND ${ARGC} GREATER 4)
    # get_filename_component(tmp ${${_var}} DIRECTORY)
    # get_filename_component(tmp_name ${tmp} NAME)
    # # if(${tmp_name} STREQUAL "include")
    # #     get_filename_component(tmp ${tmp} DIRECTORY)
    # # endif()
    # set(${ARGV4} ${tmp})
    # unset(tmp)
    # unset(tmp_name)
    # endif()
endmacro()

macro(PackageInfo _name _found _include_dir _libraries)
    if(${_found})
        SET(LIB_MSG "Yes")

        if(NOT "${_include_dir}" STREQUAL "")
            SET(LIB_MSG "Yes, at ${_include_dir}")
        endif()
    else()
        SET(LIB_MSG "No")
    endif()

    # Create a padding string to align the console output
    string(LENGTH ${_name} SIZE)
    math(EXPR SIZE 50-${SIZE})
    string(RANDOM LENGTH ${SIZE} ALPHABET " " padding)
    message(STATUS "Package ${_name} ${padding} ${LIB_MSG}")
endmacro()

macro(PackageHelper _name _found _include_dir _libraries)
    if(${_found})
        SET(LIB_MSG "Yes")
        SET(LIBS ${LIBS} ${_libraries})

        if(NOT "${_include_dir}" STREQUAL "")
            # include_directories(${_include_dir})
            SET(PACKAGE_INCLUDES ${PACKAGE_INCLUDES} ${_include_dir})
            SET(LIB_MSG "Yes, at ${_include_dir}")
        endif()
    else()
        SET(LIB_MSG "No")
    endif()

    # Create a padding string to align the console output
    string(LENGTH ${_name} SIZE)
    math(EXPR SIZE 50-${SIZE})
    string(RANDOM LENGTH ${SIZE} ALPHABET " " padding)
    message(STATUS "Package ${_name} ${padding} ${LIB_MSG}")
endmacro()

macro(PackageHelperTarget _target _found)
    if(TARGET ${_target})
        SET(LIB_MSG "Yes")
        SET(LIB_TARGETS ${LIB_TARGETS} ${_target})
        set(${_found} 1)
        get_target_property(tmp_interface_includes ${_target} INTERFACE_INCLUDE_DIRECTORIES)

        if(tmp_interface_includes AND NOT "${tmp_interface_includes}" STREQUAL "")
            SET(LIB_MSG "Yes, at ${tmp_interface_includes}")
            SET(PACKAGE_INCLUDES ${PACKAGE_INCLUDES} ${tmp_interface_includes})
        endif()
    else()
        SET(LIB_MSG "No")
        set(${_found} 0)
    endif()

    # Create a padding string to align the console output
    string(LENGTH ${_target} SIZE)

    # message("SIZE: ${SIZE}")
    math(EXPR SIZE 50-${SIZE})
    string(RANDOM LENGTH ${SIZE} ALPHABET " " padding)
    message(STATUS "Package ${_target} ${padding} ${LIB_MSG}")
endmacro()

# ###################################################################################
macro(DefaultBuildType _default_value)
    # Set a default build type if none was specified
    set(default_build_type ${_default_value})

    # Create build type drop down
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
        set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)

        # Set the possible values of build type for cmake-gui
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()

    message(STATUS "\nBuild Type: ${CMAKE_BUILD_TYPE}")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_definitions(-DCMAKE_DEBUG)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_definitions(-DCMAKE_RELEASE)
    elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        add_definitions(-DCMAKE_MINSIZEREL)
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_definitions(-DCMAKE_RELWITHDEBINFO)
    else()
        add_definitions(-DCMAKE_RELWITHDEBINFO)
    endif()
endmacro()
