cmake_minimum_required(VERSION 3.5)
project(gndlo)

## Set CXX optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic -Wall -Wextra")
## -mtune=native -march=native

set(CMAKE_CXX_STANDARD 14) # We require C++ 14 or later
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
   set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
else(CMAKE_BUILD_TYPE STREQUAL "Release")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif(CMAKE_BUILD_TYPE STREQUAL "Release")
message(STATUS "Flags: " ${CMAKE_CXX_FLAGS})

if(1)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif(1)


# ROS
# --------------------------------------
# ROS packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)


# DEPENDENCIES
# --------------------------------------
#Looking for OpenCV
find_package( OpenCV REQUIRED )

#Looking for Eigen
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

#Looking for Ceres
find_package(Ceres REQUIRED)


# INCLUDE
# --------------------------------------
include_directories(
 include/${PROJECT_NAME}
)


# CREATE LIBRARY
# --------------------------------------
file( GLOB LIB_SOURCES
${PROJECT_SOURCE_DIR}/src/Gndlo.cpp
${PROJECT_SOURCE_DIR}/src/Gndlo_Lidar.cpp
${PROJECT_SOURCE_DIR}/src/Gndlo_selection.cpp
${PROJECT_SOURCE_DIR}/src/Gndlo_planarity.cpp
${PROJECT_SOURCE_DIR}/src/Gndlo_solution.cpp
${PROJECT_SOURCE_DIR}/src/Gndlo_features.cpp
${PROJECT_SOURCE_DIR}/src/Gndlo_ground.cpp)

file( GLOB LIB_HEADERS ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/*.h )


# CREATE EXECUTABLES
# --------------------------------------
set(ODOM_EXECUTABLE "${PROJECT_NAME}_odom")
add_executable(${ODOM_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/main/Gndlo_main.cpp ${LIB_SOURCES} ${LIB_HEADERS})
set_target_properties(${ODOM_EXECUTABLE} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# TARGET AND LINK
# --------------------------------------
#Target msgs
ament_target_dependencies( ${ODOM_EXECUTABLE}
		rclcpp
		sensor_msgs
		std_msgs
		geometry_msgs
		nav_msgs
		message_filters
		cv_bridge)

#Use OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${ODOM_EXECUTABLE} ${OpenCV_LIBS} )

#Use Eigen
ament_target_dependencies( ${ODOM_EXECUTABLE} Eigen3)
ament_export_dependencies(eigen3_cmake_module)
ament_export_dependencies(Eigen3)
include_directories(include ${EIGEN3_INCLUDE_DIR})

#Use Ceres
include_directories(${CERES_INCLUDE_DIRS})
target_link_libraries(${ODOM_EXECUTABLE} ${CERES_LIBRARIES})


# INSTALL
# --------------------------------------
install(TARGETS ${ODOM_EXECUTABLE}
	ARCHIVE DESTINATION lib/${PROJECT_NAME}
    LIBRARY DESTINATION lib/${PROJECT_NAME}
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})


# PACKAGE
# --------------------------------------
ament_package()
