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
    
    
    std::string model_path = ament_index_cpp::get_package_share_directory("stereo_fusion_node") + "/models/yolov5nu.onnx";
    
    loadONNX();

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
    // object detection on left-rectified
    std::vector<Detection> detections = detect(left_rect);
    RCLCPP_INFO(get_logger(), "Detected %zu objects", detections.size());
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


std::vector<Detection> StereoFusionNode::detect(const cv::Mat& image, int input_width_, int input_height_, float conf_thresh_, float nms_thresh_)
{
    CV_Assert(!image.empty());
    CV_Assert(image.type() == CV_8UC3);

    cv::Mat blob;
    cv::dnn::blobFromImage(
        image, blob, 1.0 / 255.0,
        cv::Size(input_width_, input_height_),
        cv::Scalar(), true, false);

    RCLCPP_INFO(get_logger(), "Blob created for YOLO input.");
    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    RCLCPP_INFO(get_logger(), "YOLO inference completed.");

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    const int rows = outputs[0].size[1];
    const int dimensions = outputs[0].size[2];
    float* data = (float*)outputs[0].data;

    float x_factor = image.cols / (float)input_width_;
    float y_factor = image.rows / (float)input_height_;

    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= conf_thresh_) {
            float* class_scores = data + 5;
            cv::Mat scores(1, dimensions - 5, CV_32FC1, class_scores);
            cv::Point class_id;
            double max_score;
            cv::minMaxLoc(scores, 0, &max_score, 0, &class_id);

            if (max_score > conf_thresh_) {
                float cx = data[0];
                float cy = data[1];
                float w  = data[2];
                float h  = data[3];

                int left   = int((cx - 0.5f * w) * x_factor);
                int top    = int((cy - 0.5f * h) * y_factor);
                int width  = int(w * x_factor);
                int height = int(h * y_factor);

                class_ids.push_back(class_id.x);
                confidences.push_back(confidence * max_score);
                boxes.emplace_back(left, top, width, height);
            }
        }
        data += dimensions;
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, indices);

    std::vector<Detection> detections;
    for (int idx : indices) {
        detections.push_back({
            class_ids[idx],
            confidences[idx],
            boxes[idx]
        });
    }

    return detections;
}


void StereoFusionNode::loadONNX()
{
    std::string model_path =
        ament_index_cpp::get_package_share_directory("stereo_fusion_node") +
        "/models/yolov5n.onnx";

    net_ = cv::dnn::readNetFromONNX(model_path);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoFusionNode>());
    rclcpp::shutdown();
    return 0;
}
