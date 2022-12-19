
CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

#
# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take their definition
# and dump it in the cache along with proper documentation, otherwise set CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
#


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)

if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
       add_compile_options (-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
       add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()



SET(CMAKE_BUILD_TYPE Debug )

PROJECT(hglmbslam)




find_package(catkin REQUIRED COMPONENTS
  roscpp
  visualization_msgs
  sensor_msgs
  tf2_ros
)
#find_package(GTSAM REQUIRED)


###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include  ${OpenCV_INCLUDE_DIRS}
  CATKIN_DEPENDS roscpp   std_msgs   sensor_msgs navigation_msgs visualization_msgs geometry_msgs
  DEPENDS
)



# For cmake custom library searches
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
MESSAGE("adding path to cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake " )
# choose latest opengl
set(OpenGL_GL_PREFERENCE GLVND)
# Compiler settings
# Note: to change compiler, use for example...
# cmake -D CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=c++ .
#SET(CMAKE_C_COMPILER "gcc")
#SET(CMAKE_CXX_COMPILER "g++")



option (BUILD_WITH_MARCH_NATIVE "Build with \"-march native\"" OFF)
  if(BUILD_WITH_MARCH_NATIVE AND NOT "${CMAKE_SYSTEM_PROCESSOR}" MATCHES "arm" AND "${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
  endif()




if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   add_compile_options (-fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   add_compile_options (-fcolor-diagnostics)
endif ()
#MESSAGE("CMAKE_CXX_COMPILTER_ID = ${CMAKE_CXX_COMPILER_ID}")
IF(APPLE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-qualifiers ") # For Eigen const warnings
  IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    MESSAGE("NOTE: Libraries being linked to RFSSLAM must also be compiled using GCC")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-enum-compare ") # For Eigen enum compare warnings
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-q ") #required for march=native errors. This forces clang assembler to be used, and will produce some argument unused warnings. 
  ENDIF()
ENDIF()




 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -D_GLIBCXX_ASSERTIONS")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -DLLVM_ENABLE_EXPENSIVE_CHECKS=ON")
# sanitizer

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   -fsanitize=address  -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fsanitize=address  -fno-omit-frame-pointer")

if (USE_SANITIZER STREQUAL "Address")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address  -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address  -fno-omit-frame-pointer")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}  -fsanitize=address  -fno-omit-frame-pointer")
elseif (USE_SANITIZER STREQUAL "Thread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}  -fsanitize=thread")
elseif (USE_SANITIZER STREQUAL "Undefined")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}  -fsanitize=undefined")
elseif (USE_SANITIZER STREQUAL "Leak")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}  -fsanitize=leak")
elseif (USE_SANITIZER STREQUAL "Memory")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=memory  -fno-omit-frame-pointer -fsanitize-memory-track-origins")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory  -fno-omit-frame-pointer -fsanitize-memory-track-origins")
    set (CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS}  -fsanitize=memory  -fno-omit-frame-pointer -fsanitize-memory-track-origins")
else()
    SET(USE_SANITIZER "disabled" CACHE STRING "set sanitizer type.")
endif()
    
    
    # Google perftools
FIND_PACKAGE(Perftools)
IF(Perftools_FOUND)
  IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(USE_CPU_PROFILER OFF CACHE  BOOL "Use Google Perftools for CPU profiling.")
    SET(USE_HEAP_PROFILER OFF CACHE BOOL "Use Google Perftools for heap profiling.")
  ELSE()
    SET(USE_CPU_PROFILER OFF) # Does not play nice with Clang
    SET(USE_HEAP_PROFILER OFF)
  ENDIF()
  IF(USE_CPU_PROFILER)
    message("enabling cpu profiler ${USE_CPU_PROFILER}")
    ADD_DEFINITIONS(-D_PERFTOOLS_CPU)
    IF(APPLE)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no_pie ") # Required due to ASLR
    ENDIF()
  ENDIF()
  IF(USE_HEAP_PROFILER)
    message("enabling heap profiler ${USE_HEAP_PROFILER}")
    ADD_DEFINITIONS(-D_PERFTOOLS_HEAP)
    IF(APPLE)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no_pie ") # Required due to ASLR
    ENDIF()
  ENDIF()
ENDIF()



SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")
# OpenMP
# IF(APPLE AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
#   SET(USE_OPENMP OFF) # OpenMP does not yet work with Clang, but hopefully soon
# ELSE()
#   SET(USE_OPENMP OFF CACHE BOOL "Use OpenMP to compile multithreaded versions of SLAM algorithms.")
# ENDIF()
# IF(USE_OPENMP)
#   IF("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
#     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libomp ")
#   ELSE()
#     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp ")
#   ENDIF()
# ENDIF()


#Threads

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)




# Eigen
message("looking for eigen")
FIND_PACKAGE(Eigen3 REQUIRED)
message("looking for eigen")

find_package(PCL REQUIRED)
# Boost
FIND_PACKAGE(Boost REQUIRED COMPONENTS timer chrono system filesystem graph program_options container )
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON) 
set(Boost_USE_STATIC_RUNTIME ON)




# Google Ceres

FIND_PACKAGE(Sophus REQUIRED)

# G2O 
FIND_PACKAGE(g2o REQUIRED)
FIND_PACKAGE(CSparse REQUIRED)
FIND_PACKAGE(yaml-cpp REQUIRED)
MESSAGE("yaml-cpp: ${YAML_CPP_FOUND}  , ${YAML_CPP_INCLUDE_DIR} , ${YAML_CPP_LIBRARIES}")


find_package (glog  REQUIRED)
#  Opencv
find_package(OpenCV 4.4)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV > 4.4 not found.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})



    



FILE(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples")
FILE(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/test")




#ADD_SUBDIRECTORY()
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(src)
INCLUDE_DIRECTORIES(test)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIRS})
message("eigendirs: ${EIGEN_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(${PERFTOOLS_INCLUDE_DIRS})


INCLUDE_DIRECTORIES(${G2O_INCLUDE_DIRS})


INCLUDE_DIRECTORIES(${CSPARSE_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIRS})
include_directories(${YAML_CPP_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${catkin_INCLUDE_DIRS})
message("===========")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
message("===========")
include_directories(${PCL_INCLUDE_DIRS})



message("===========")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
message("===========")

ADD_LIBRARY(rfsslam 
  src/GaussianGenerators.cpp
  src/AssociationSampler.cpp
  src/OrbslamMapPoint.cpp
  src/OrbslamPose.cpp
  )
  
  
TARGET_LINK_LIBRARIES(rfsslam   ${YAML_CPP_LIBRARIES}  ${Boost_LIBRARIES}  ${OpenCV_LIBS} ${Sophus_LIBRARIES} fmt::fmt  ${catkin_LIBRARIES} 
${PCL_LIBRARIES}    )


link_directories(${PCL_LIBRARY_DIRS})





ADD_EXECUTABLE(vglmbslam6d src/vglmbslam6d_euroc.cpp  src/external/ORBextractor.cc src/external/Converter.cc)
# TARGET_LINK_LIBRARIES(vglmbslam6d  ${YAML_CPP_LIBRARIES} glog::glog ${Boost_LIBRARIES} rfsslam  ${CSPARSE_LIBRARY} g2o_core_d g2o_csparse_extension_d g2o_solver_csparse_d g2o_types_slam3d_d g2o_types_sba_d g2o_types_sim3_d  g2o_stuff_d   
# ${OpenCV_LIBS} ${Sophus_LIBRARIES}  ${catkin_LIBRARIES} ${PCL_LIBRARIES}  )



# TARGET_LINK_LIBRARIES(vglmbslam6d  yaml-cpp  ${Boost_LIBRARIES} rfsslam ${CERES_LIBRARIES} ${CSPARSE_LIBRARY} g2o_core g2o_csparse_extension g2o_solver_csparse g2o_types_slam3d g2o_types_sba g2o_types_sim3  g2o_stuff  
# ${OpenCV_LIBS} ${Sophus_LIBRARIES}  ${catkin_LIBRARIES} ${PCL_LIBRARIES}   )

TARGET_LINK_LIBRARIES(vglmbslam6d  ${YAML_CPP_LIBRARIES} glog::glog  ${Boost_LIBRARIES} rfsslam  ${CSPARSE_LIBRARY} g2o_core_rd g2o_csparse_extension_rd g2o_solver_csparse_rd g2o_types_slam3d_rd g2o_types_sba_rd g2o_types_sim3_rd  g2o_stuff_rd  
${OpenCV_LIBS} ${Sophus_LIBRARIES}  ${catkin_LIBRARIES} ${PCL_LIBRARIES}   )


