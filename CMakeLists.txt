cmake_minimum_required(VERSION 3.18)
project(brainskel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address,undefined")

# ---- OpenMP ----------------------------------------------------------------
find_package(OpenMP REQUIRED)

# ---- ZLIB ------------------------------------------------------------------
find_package(ZLIB REQUIRED)

# ---- voro++ ----------------------------------------------------------------
# The upstream repo uses a Makefile, so we fetch and build manually.
include(FetchContent)

FetchContent_Declare(
  voro
  GIT_REPOSITORY https://github.com/chr1shr/voro.git
  GIT_TAG        master
  GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(voro)
if(NOT voro_POPULATED)
  FetchContent_Populate(voro)
endif()

# voro++.cc is a unity build that #includes all the other .cc files.
# cmd_line.cc is a standalone executable; do not include it.
add_library(voro_lib STATIC "${voro_SOURCE_DIR}/src/voro++.cc")
target_include_directories(voro_lib PUBLIC "${voro_SOURCE_DIR}/src")
# suppress warnings from third-party code
target_compile_options(voro_lib PRIVATE -O3 -w)

# ---- brainskel library -----------------------------------------------------
add_library(brainskel_lib STATIC
  src/mesh.cpp
  src/sampling.cpp
  src/voronoi.cpp
)
target_include_directories(brainskel_lib PUBLIC include)
target_link_libraries(brainskel_lib PUBLIC voro_lib OpenMP::OpenMP_CXX ZLIB::ZLIB)
target_compile_options(brainskel_lib PRIVATE -Wall -Wextra)

# ---- tools -----------------------------------------------------------------
foreach(tool sample_points medial_axis)
  add_executable(${tool} tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE brainskel_lib)
endforeach()

# ---- install ---------------------------------------------------------------
install(TARGETS sample_points medial_axis DESTINATION bin)
