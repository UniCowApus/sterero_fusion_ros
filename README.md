Stereo Fusion Node
Overview

stereo_fusion_node is a ROS 2 Humble package that performs real time stereo vision fusion. I got it working about 2 hours before the deadline (: ). It synchronizes stereo images, rectifies them using camera calibration data, computes a dense disparity map, runs object detection using a YOLOv5 ONNX model, and estimates 3D object positions in the camera frame.

The node publishes annotated images, disparty visualizations, and 3D detection markers for use in RViz and downstream perception pipelines. In the future, I plan on creating a ros node that uses [Sonos-ESP32](https://github.com/javos65/Sonos-ESP32) to dynamically adjust the volume of speakers as I walk around my house. Maybe I'll have that done in time for my presentation!

Features

Stereo image synchronization and rectification
Dense disparity estimation using OpenCV StereoSGBM
YOLOv5 object detection via OpenCV DNN
Per object depth estimation from disparity
2D and 3D visualization outputs

Package Structure
stereo_fusion_node
├── include/stereo_fusion_node/stereo_fusion_node.hpp

├── src/stereo_fusion_node.cpp

├── launch/stereo_fusion_node.launch.py

├── models/yolov5n.onnx

├── CMakeLists.txt

└── package.xml

Topics
Subscribed

/left/left_camera_node/image_raw/compressed
/right/right_camera_node/image_raw/compressed
/left/left_camera_node/camera_info
/right/right_camera_node/camera_info

Published

/detections/image
/disparity
/detections/markers

Running the Node
ros2 launch stereo_fusion_node stereo_fusion_node.launch.py

Dependencies

rclcpp
sensor_msgs
visualization_msgs
message_filters
cv_bridge
image_geometry
OpenCV
ament_index_cpp

Notes

My documenttion is lacking, and I'd prefer not to require recompiling openCV to 4.14 to enable ONNX models, but my camera only arrived 24 hours before submission time, so there is certainly room for improvement. If there are any further questions/suggestions, please email me below.

Author

David Rogers
david.c.rogers488@gmail.com
