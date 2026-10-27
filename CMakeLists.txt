cmake_minimum_required(VERSION 3.20)

project(hglmbslam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# hardware popcount for ORB descriptor distances (std::bitset::count)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  add_compile_options(-mpopcnt)
endif()

option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if(FORCE_COLORED_OUTPUT)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
  endif()
endif()

if(USE_SANITIZER STREQUAL "Address")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
elseif(USE_SANITIZER STREQUAL "Thread")
  add_compile_options(-fsanitize=thread)
  add_link_options(-fsanitize=thread)
elseif(USE_SANITIZER STREQUAL "Undefined")
  add_compile_options(-fsanitize=undefined)
  add_link_options(-fsanitize=undefined)
elseif(USE_SANITIZER STREQUAL "Leak")
  add_compile_options(-fsanitize=leak)
  add_link_options(-fsanitize=leak)
elseif(USE_SANITIZER STREQUAL "Memory")
  add_compile_options(-fsanitize=memory -fno-omit-frame-pointer -fsanitize-memory-track-origins)
  add_link_options(-fsanitize=memory)
else()
  set(USE_SANITIZER "disabled" CACHE STRING "set sanitizer type.")
endif()

find_package(Perftools)
if(Perftools_FOUND AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(USE_CPU_PROFILER OFF CACHE BOOL "Use Google Perftools for CPU profiling.")
  set(USE_HEAP_PROFILER OFF CACHE BOOL "Use Google Perftools for heap profiling.")
  if(USE_CPU_PROFILER)
    add_compile_definitions(_PERFTOOLS_CPU)
  endif()
  if(USE_HEAP_PROFILER)
    add_compile_definitions(_PERFTOOLS_HEAP)
  endif()
endif()

# ROS2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(OpenMP REQUIRED)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

cmake_policy(SET CMP0167 NEW)
find_package(Boost CONFIG REQUIRED COMPONENTS timer chrono filesystem graph container program_options)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(GTSAM 4.3 REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbbmalloc_proxy)
find_package(OpenCV 4.4 REQUIRED)

add_library(rfsslam
  src/GaussianGenerators.cpp
  src/AssociationSampler.cpp
  src/OrbslamMapPoint.cpp
  src/OrbslamPose.cpp
  src/external/ORBextractor.cc
)
target_include_directories(rfsslam
  PUBLIC
    ${GTSAM_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(rfsslam
  PUBLIC
    gtsam
    Eigen3::Eigen
    Sophus::Sophus
    yaml-cpp::yaml-cpp
    fmt::fmt
    glog::glog
    ${OpenCV_LIBS}
    Boost::timer Boost::chrono Boost::filesystem Boost::graph Boost::container
    OpenMP::OpenMP_CXX
    Threads::Threads
)

add_executable(vglmbslam6d src/vglmbslam6d_euroc.cpp)
target_link_libraries(vglmbslam6d
  PRIVATE
    rfsslam
    rclcpp::rclcpp
    ${visualization_msgs_TARGETS}
    ${geometry_msgs_TARGETS}
    Boost::program_options
    ${Perftools_PROFILER_LIBRARY}
    TBB::tbbmalloc_proxy  # replaces malloc/new process-wide; ~4% faster than glibc malloc here
)

install(TARGETS rfsslam vglmbslam6d
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY cfg launch rviz DESTINATION share/${PROJECT_NAME}/)

ament_package()
