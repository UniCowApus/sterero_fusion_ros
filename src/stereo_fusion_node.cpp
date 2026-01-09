#include "stereo_fusion_node/stereo_fusion_node.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

StereoFusionNode::StereoFusionNode()
: Node("stereo_fusion_node")
{
  left_image_sub_.subscribe(this, "/stereo/left/image_raw");
  right_image_sub_.subscribe(this, "/stereo/right/image_raw");

  left_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "/stereo/left/camera_info", 10,
      std::bind(&StereoFusionNode::leftInfoCallback, this, _1));

  right_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "/stereo/right/camera_info", 10,
      std::bind(&StereoFusionNode::rightInfoCallback, this, _1));

  sync_ = std::make_shared<Synchronizer>(
      SyncPolicy(10), left_image_sub_, right_image_sub_);

  sync_->registerCallback(
      std::bind(&StereoFusionNode::stereoCallback, this, _1, _2));

  RCLCPP_INFO(get_logger(), "Stereo fusion pipeline initialized.");
}

void StereoFusionNode::leftInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  left_cam_model_.fromCameraInfo(msg);
  left_info_received_ = true;
}

void StereoFusionNode::rightInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  right_cam_model_.fromCameraInfo(msg);
  right_info_received_ = true;
}

void StereoFusionNode::stereoCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &left_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &right_msg)
{
  if (!left_info_received_ || !right_info_received_) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Waiting for cam calib message");
    return;
  }

  cv::Mat left_rect, right_rect;

  if (!rectifyStereoPair(left_msg, right_msg, left_rect, right_rect)) {
    RCLCPP_ERROR(get_logger(), "Rectification failed");
    return;
  }else{
    RCLCPP_INFO(get_logger(), "Stereo rectification succeeded.");
  }

}

bool StereoFusionNode::rectifyStereoPair(
    const sensor_msgs::msg::Image::ConstSharedPtr &left_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &right_msg,
    cv::Mat &left_rectified,
    cv::Mat &right_rectified)
{
  cv_bridge::CvImageConstPtr left_cv;
  cv_bridge::CvImageConstPtr right_cv;

  try {
    left_cv  = cv_bridge::toCvShare(left_msg, "bgr8");
    right_cv = cv_bridge::toCvShare(right_msg, "bgr8");
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
    return false;
  }

  left_cam_model_.rectifyImage(left_cv->image, left_rectified);
  right_cam_model_.rectifyImage(right_cv->image, right_rectified);

  return true;
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StereoFusionNode>());
  rclcpp::shutdown();
  return 0;
}
