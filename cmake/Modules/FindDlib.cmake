# Once done these will be defined:
#
#  DLIB_FOUND
#  DLIB_INCLUDE_DIRS
#  DLIB_LIBRARIES
#


if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_lib_suffix win64)
else()
        set(_lib_suffix win32)
endif()

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(_build_type Debug)
else()
    set(_build_type Release)
endif()

find_path(dlib_INCLUDE_DIR
        NAMES dlib
        PATHS
                ${CMAKE_SOURCE_DIR}/deps
        PATH_SUFFIXES
                dlib)

find_library(dlib_LIB
        NAMES dlib19.20.99_release_64bit_msvc1916
        PATHS
              ${CMAKE_SOURCE_DIR}/deps/dlib/${_build_type}/${_lib_suffix}
        PATH_SUFFIXES
                lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Dlib DEFAULT_MSG dlib_LIB dlib_INCLUDE_DIR)
mark_as_advanced(dlib_LIB dlib_LIB)

if(DLIB_FOUND)
        set(DLIB_INCLUDE_DIRS ${dlib_INCLUDE_DIR})
        set(DLIB_LIBRARIES ${dlib_LIB})
endif()
