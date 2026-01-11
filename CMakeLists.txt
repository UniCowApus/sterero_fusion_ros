cmake_minimum_required(VERSION 3.8)
project(stereo_fusion_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_geometry REQUIRED)
find_package(OpenCV 4.14 REQUIRED) #need to manually compile this with ros Humble, but required for onnx


include_directories(
  include
)

add_executable(stereo_fusion_node
  src/stereo_fusion_node.cpp
)

ament_target_dependencies(stereo_fusion_node
  rclcpp
  sensor_msgs
  message_filters
  cv_bridge
  image_geometry
  visualization_msgs
)

target_link_libraries(stereo_fusion_node
  ${OpenCV_LIBRARIES}
)

install(TARGETS stereo_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
