# Once done these will be defined:
#
#  Torch_INCLUDE_DIRS
#  Torch_LIBRARIES


find_package(PkgConfig QUIET)


if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # 64bit
    if (CMAKE_BUILD_TYPE MATCHES "Debug")
        set(Torch_DIR ${CMAKE_SOURCE_DIR}/deps/libtorch/Debug/share/cmake/Torch)
    else()
        set(Torch_DIR ${CMAKE_SOURCE_DIR}/deps/libtorch/Release/share/cmake/Torch)
    endif()
else()
    # 32bit
    message(FATAL_ERROR "pelease provide 32bit torch lib!")
endif()

message(${Torch_DIR})
find_package(Torch REQUIRED)


set(Torch_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
set(Torch_LIBRARIES ${TORCH_LIBRARIES})

