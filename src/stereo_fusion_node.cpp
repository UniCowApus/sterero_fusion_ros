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
    if (!left_info_received_ || !right_info_received_)
    {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Waiting for cam calib message");
        return;
    }
    // rectify
    cv::Mat left_rect, right_rect;

    if (!rectifyStereoPair(left_msg, right_msg, left_rect, right_rect))
    {
        RCLCPP_ERROR(get_logger(), "Rectification failed");
        return;
    }
    else
    {
        RCLCPP_INFO(get_logger(), "Stereo rectification succeeded.");
    }
    // disparty with greyscale
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_rect, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_rect, right_gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat disparity;
    if (!computeDisparitySGBM(left_gray, right_gray, disparity))
    {
        RCLCPP_ERROR(get_logger(), "Disparity computation failed");
        return;
    }
    else
    {
        RCLCPP_INFO(get_logger(), "Disparity computation succeeded.");
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

    try
    {
        left_cv = cv_bridge::toCvShare(left_msg, "bgr8");
        right_cv = cv_bridge::toCvShare(right_msg, "bgr8");
    }
    catch (const cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(get_logger(), "another cv_bridge error: %s", e.what());
        return false;
    }

    left_cam_model_.rectifyImage(left_cv->image, left_rectified);
    right_cam_model_.rectifyImage(right_cv->image, right_rectified);

    return true;
}

bool StereoFusionNode::computeDisparitySGBM(
    const cv::Mat &left_rect,
    const cv::Mat &right_rect,
    cv::Mat &disparity)
{
    CV_Assert(left_rect.type() == CV_8UC1);
    CV_Assert(right_rect.type() == CV_8UC1);
    CV_Assert(left_rect.size() == right_rect.size());

    int minDisparity = 0;
    int numDisparities = 64;
    int blockSize = 9;
    int P1 = 8 * blockSize * blockSize;
    int P2 = 32 * blockSize * blockSize;

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        minDisparity,
        numDisparities,
        blockSize,
        P1,
        P2,
        1,  // disp12maxDif
        10, // preFilterCap
        50, // uniquenessRatio
        0,  // speckleWindwSize - don't need I guess
        0,  // speckleRange -Same as above
        cv::StereoSGBM::MODE_SGBM);

    cv::Mat disparity_s16;
    try
    {
        sgbm->compute(left_rect, right_rect, disparity_s16);
    }
    catch (const cv::Exception &e)
    {
        RCLCPP_WARN(this->get_logger(), "StereoSGBM didn't compute %s", e.what());
        return false;
    }
    disparity_s16.convertTo(disparity, CV_32F, 1.0 / 16.0);
    return true;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoFusionNode>());
    rclcpp::shutdown();
    return 0;
}
