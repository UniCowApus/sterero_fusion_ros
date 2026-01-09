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

class StereoFusionNode : public rclcpp::Node
{
public:
  StereoFusionNode();

private:

  void leftInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void rightInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void stereoCallback(const sensor_msgs::msg::Image::ConstSharedPtr &left_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr &right_msg);
  bool rectifyStereoPair(const sensor_msgs::msg::Image::ConstSharedPtr &left_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr &right_msg,
      cv::Mat &left_rectified,
      cv::Mat &right_rectified);

  message_filters::Subscriber<sensor_msgs::msg::Image> left_image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> right_image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_sub_;
  
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<
          sensor_msgs::msg::Image,
          sensor_msgs::msg::Image>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;
  std::shared_ptr<Synchronizer> sync_;

  image_geometry::PinholeCameraModel left_cam_model_;
  image_geometry::PinholeCameraModel right_cam_model_;

  bool left_info_received_  = false;
  bool right_info_received_ = false;
};
