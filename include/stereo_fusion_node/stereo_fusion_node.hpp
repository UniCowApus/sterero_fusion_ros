#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "visualization_msgs/msg/marker_array.hpp"
#include <sensor_msgs/msg/compressed_image.hpp>

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

struct Detection3D
{
    Detection det;
    cv::Point3f position_cam;
    float disparity;
};



class StereoFusionNode : public rclcpp::Node
{
public:
  StereoFusionNode();

private:

  void leftInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void rightInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void stereoCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &left_msg,
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr &right_msg);
  bool rectifyStereoPair(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &left_msg,
      const sensor_msgs::msg::CompressedImage::ConstSharedPtr &right_msg,
      cv::Mat &left_rectified,
      cv::Mat &right_rectified);
  bool computeDisparitySGBM(
    const cv::Mat &left_rect,
    const cv::Mat &right_rect,
    cv::Mat &disparity);
  std::vector<Detection> detect(const cv::Mat& image, int input_width_ = 640, int input_height_ = 640, float conf_thresh_ = 0.4f, float nms_thresh_ = 0.5f); 
  void loadONNX();
  bool estimateDepthFromDisparity(
    const std::vector<Detection>& detections,
    const cv::Mat& disparity,
    std::vector<Detection3D>& out_detections_3d);

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  message_filters::Subscriber<sensor_msgs::msg::CompressedImage> left_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CompressedImage> right_image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_sub_;
  
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<
          sensor_msgs::msg::CompressedImage,
          sensor_msgs::msg::CompressedImage>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;
  std::shared_ptr<Synchronizer> sync_;

  image_geometry::PinholeCameraModel left_cam_model_;
  image_geometry::PinholeCameraModel right_cam_model_;

  bool left_info_received_  = false;
  bool right_info_received_ = false;
  cv::dnn::Net net_;

};
