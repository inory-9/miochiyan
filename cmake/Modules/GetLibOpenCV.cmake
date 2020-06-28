# Once done these will be defined:
#
#  OpenCV_INCLUDE_DIRS_M
#  OpenCV_LIBRARIES


find_package(PkgConfig QUIET)


if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # 64bit
    if (CMAKE_BUILD_TYPE MATCHES "Debug")
        set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/deps/opencv/Debug)
    else()
        set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/deps/opencv/Release)
    endif()
else()
    # 32bit
    message(FATAL_ERROR "pelease provide 32bit opencv lib!")
endif()

find_package(OpenCV REQUIRED)


set(OpenCV_INCLUDE_DIRS_M ${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBRARIES ${OpenCV_LIBS})
