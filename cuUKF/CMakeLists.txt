cmake_minimum_required(VERSION 3.8)

get_filename_component(PACKAGE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message("Creating Package: ${PACKAGE_NAME}")
project(${PACKAGE_NAME} LANGUAGES CXX CUDA)  ## this package name is the name of the directory this cmake file is in

add_compile_definitions(CMAKE_PACKAGE_NAME=${PACKAGE_NAME})

# Default to C++17 for humble
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-unknown-pragmas")


set(ROS_DEPENDS
    rclcpp
    std_msgs
    ament_cmake
    sensor_msgs
    nav_msgs
    message_filters
    CUDAToolkit
  )

# find dependencies
foreach(DEPEND ${ROS_DEPENDS})
  find_package(${DEPEND} REQUIRED)
endforeach(DEPEND)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/cuUKF)
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

message("buliding ${PROJECT_NAME} node: ")

add_library(cpu_filter
  src/cpuFilter.cpp
)

add_executable(cpu_node nodes/cpuNode.cpp)
ament_target_dependencies(cpu_node
  rclcpp
  sensor_msgs
  nav_msgs
  std_msgs
  Eigen3
  message_filters
  geometry_msgs
)

ament_target_dependencies(cpu_filter
  rclcpp
  sensor_msgs
  nav_msgs
  std_msgs
  Eigen3
  message_filters
  geometry_msgs
)

target_link_libraries(cpu_node
  cpu_filter
)

#############################################
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

add_library(gpu_filter
  src/gpuFilter.cu
)

target_include_directories(gpu_filter PUBLIC
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(gpu_filter
  CUDA::cublas
  CUDA::cusolver
  CUDA::cuda_driver
)

set_target_properties(gpu_filter PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(gpu_filter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unknown-pragmas>)


add_executable(core_node nodes/coreNode.cpp)
ament_target_dependencies(core_node
  rclcpp
  sensor_msgs
  nav_msgs
  std_msgs
  message_filters
  geometry_msgs
)

ament_target_dependencies(gpu_filter
  rclcpp
  sensor_msgs
  nav_msgs
  std_msgs
  message_filters
  geometry_msgs
)

target_link_libraries(core_node
  gpu_filter
)

###############################################
install(TARGETS
  cpu_node
  core_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
