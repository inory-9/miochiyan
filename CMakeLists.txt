cmake_minimum_required(VERSION 3.16)

project(miochiyan LANGUAGES CXX)

option(BUILD_CAPTIONS "Build captions" FALSE)

set(CMAKE_DEBUG_POSTFIX "d" )
set(CMAKE_RELEASE_POSTFIX "")
set(CMAKE_RELWITHDEBINFO_POSTFIX "rd")
set(CMAKE_MINSIZEREL_POSTFIX "s")


if (CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_BUILD_POSTFIX "${CMAKE_RELEASE_POSTFIX}")
elseif (CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
    set(CMAKE_BUILD_POSTFIX "${CMAKE_MINSIZEREL_POSTFIX}")
elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    set(CMAKE_BUILD_POSTFIX "${CMAKE_RELWITHDEBINFO_POSTFIX}")
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_BUILD_POSTFIX "${CMAKE_DEBUG_POSTFIX}")
else()
    set(CMAKE_BUILD_POSTFIX "")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(CMAKE_AUTOUIC ON)
#set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
        add_definitions(-DUNICODE -D_UNICODE -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_WARNINGS)
endif()

if(MSVC)
        add_compile_options("/MP")
        set(CMAKE_C_FLAGS_DEBUG "/DDEBUG=1 /D_DEBUG=1 ${CMAKE_C_FLAGS_DEBUG}")
        set(CMAKE_CXX_FLAGS_DEBUG "/DDEBUG=1 /D_DEBUG=1 ${CMAKE_C_FLAGS_DEBUG}")

        if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SAFESEH:NO")
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /SAFESEH:NO")
                set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /SAFESEH:NO")
        endif()

        if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
                string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        else()
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
        endif()

        # Disable pointless constant condition warnings
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4127 /wd4201 /wd4456 /wd4457 /wd4458 /wd4459 /wd4595")

else()
        if(MINGW)
                set(CMAKE_WIDL "widl" CACHE STRING "wine IDL header file generation program")
                set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_WIN32_WINNT=0x0600 -DWINVER=0x0600")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_WIN32_WINNT=0x0600 -DWINVER=0x0600")
        endif()
        set(CMAKE_C_FLAGS_DEBUG "-DDEBUG=1 -D_DEBUG=1 ${CMAKE_C_FLAGS_DEBUG}")
        set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG=1 -D_DEBUG=1 ${CMAKE_C_FLAGS_DEBUG}")
endif()


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(WIN32)
        if (QTDIR OR DEFINED ENV{QTDIR} OR DEFINED ENV{QTDIR32} OR DEFINED ENV{QTDIR64})
                # Qt path set by user or env var
        else()
                set(QTDIR "" CACHE PATH "Path to Qt (e.g. C:/Qt/5.7/msvc2015_64)")
                message(WARNING "QTDIR variable is missing.  Please set this variable to specify path to Qt (e.g. C:/Qt/5.7/msvc2015_64)")
        endif()
endif()

find_package(Qt5 COMPONENTS Svg REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(Dlib REQUIRED)

include(GetLibTorch)
include(GetLibOpenCV)


set(UI_SOURCES
    ui/main.cpp
    ui/mainwindow.cpp
     )

set(UI_HEADERS
    ui/mainwindow.h
     )

set(CORE_SOURCES
    core/expression_capture.cpp
    )

set(CORE_HEADERS
    core/expression_capture.h
    )


set(UI_FILES
    ui/forms/mainwindow.ui
     )

set(QRC_FILES
     )

source_group("ui" FILES ${UI_SOURCES} ${UI_HEADERS} ${UI_FILES})
source_group("core" FILES ${CORE_SOURCES} ${CORE_HEADERS})

qt5_wrap_cpp(QT_MOC_HEADERS ${UI_HEADERS})
qt5_wrap_ui(QT_UI_HEADERS ${UI_FILES})
qt5_add_resources(QT_QRC_RESOURCES ${QRC_FILES})



add_executable(${PROJECT_NAME}
    ${UI_SOURCES}
    ${UI_HEADERS}
    ${CORE_SOURCES}
    ${CORE_HEADERS}
    ${QT_MOC_HEADERS}
    ${QT_UI_HEADERS}
    ${QT_QRC_RESOURCES}
    )

include_directories(${TORCH_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS_M}
                    ${DLIB_INCLUDE_DIRS}
                    )

target_link_libraries(${PROJECT_NAME}
                     ${TORCH_LIBRARIES}
                     ${OpenCV_LIBS}
                     ${DLIB_LIBRARIES}
                     Qt5::Widgets
                     )
