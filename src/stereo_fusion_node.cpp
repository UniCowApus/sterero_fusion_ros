#include "stereo_fusion_node/stereo_fusion_node.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

StereoFusionNode::StereoFusionNode()
    : Node("stereo_fusion_node")
{
    left_image_sub_.subscribe(this, "/left/left_camera_node/image_raw/compressed");
    right_image_sub_.subscribe(this, "/right/right_camera_node/image_raw/compressed");

    left_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/left/left_camera_node/camera_info", 10,
        std::bind(&StereoFusionNode::leftInfoCallback, this, _1));

    right_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/right/right_camera_node/camera_info", 10,
        std::bind(&StereoFusionNode::rightInfoCallback, this, _1));

    sync_ = std::make_shared<Synchronizer>(
        SyncPolicy(10), left_image_sub_, right_image_sub_);

    sync_->registerCallback(
        std::bind(&StereoFusionNode::stereoCallback, this, _1, _2));
    
    left_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("detections/image", 1);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "detections/markers", 10);
    
    std::string model_path = ament_index_cpp::get_package_share_directory("stereo_fusion_node") + "/models/yolov5nu.onnx";

    loadONNX();

    RCLCPP_INFO(get_logger(), "Stereo fusion pipeline initialized.");
}

void StereoFusionNode::leftInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{   
    if(!left_info_received_){
        left_cam_model_.fromCameraInfo(msg);
        left_info_received_ = true;
    }
}

void StereoFusionNode::rightInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{   
    if(!right_info_received_){
        right_cam_model_.fromCameraInfo(msg);
        right_info_received_ = true;
    }
}

void StereoFusionNode::stereoCallback(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr &left_msg,
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr &right_msg)
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
    RCLCPP_INFO(get_logger(), "Detected %zu objects with", detections.size());
    for (const auto &det : detections)
    {
        RCLCPP_INFO(get_logger(), "Class ID: %d, Confidence: %.2f, Box: [%d, %d, %d, %d]",
                    det.class_id, det.confidence,
                    det.box.x, det.box.y, det.box.width, det.box.height);
    }
    //depth estimate
    std::vector<Detection3D> detections_3d;
    if (!estimateDepthFromDisparity(detections, disparity, detections_3d))
    {
        RCLCPP_WARN(get_logger(), "No valid 3D detections");
        return;
    }

    // publish vizsualation
    cv::Mat display_image = left_rect.clone();
    for (const auto &d : detections_3d)
    {
        cv::rectangle(display_image, d.det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "ID:" + std::to_string(d.det.class_id) +
                            " Z:" + std::to_string(d.position_cam.z);
        cv::putText(display_image, label, cv::Point(d.det.box.x, d.det.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    sensor_msgs::msg::Image img_msg = *cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", display_image).toImageMsg();
    left_image_pub_->publish(img_msg);
    visualization_msgs::msg::MarkerArray marker_array;

    int id = 0;
    for (const auto &d : detections_3d)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = left_msg->header.frame_id;
        marker.header.stamp = this->now();
        marker.ns = "detections";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = d.position_cam.x;
        marker.pose.position.y = d.position_cam.y;
        marker.pose.position.z = d.position_cam.z;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        marker_array.markers.push_back(marker);
    }
    marker_pub_->publish(marker_array);

    // log any 3d detections
    for (const auto &d : detections_3d)
    {
        RCLCPP_INFO(get_logger(),
                    "Class %d | XYZ = [%.2f, %.2f, %.2f] m | disparity = %.2f",
                    d.det.class_id,
                    d.position_cam.x,
                    d.position_cam.y,
                    d.position_cam.z,
                    d.disparity);
    }
}

bool StereoFusionNode::rectifyStereoPair(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr &left_msg,
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr &right_msg,
    cv::Mat &left_rectified,
    cv::Mat &right_rectified)
{
    cv_bridge::CvImagePtr left_cv;
    cv_bridge::CvImagePtr right_cv;

    try
    {
        left_cv  = cv_bridge::toCvCopy(left_msg, "bgr8");
        right_cv = cv_bridge::toCvCopy(right_msg, "bgr8");
    }
    catch (const cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(get_logger(), "cv_bridge decode error: %s", e.what());
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

std::vector<Detection> StereoFusionNode::detect(const cv::Mat &image, int input_width_, int input_height_, float conf_thresh_, float nms_thresh_)
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
    float *data = (float *)outputs[0].data;

    float x_factor = image.cols / (float)input_width_;
    float y_factor = image.rows / (float)input_height_;

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= conf_thresh_)
        {
            float *class_scores = data + 5;
            cv::Mat scores(1, dimensions - 5, CV_32FC1, class_scores);
            cv::Point class_id;
            double max_score;
            cv::minMaxLoc(scores, 0, &max_score, 0, &class_id);

            if (max_score > conf_thresh_)
            {
                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((cx - 0.5f * w) * x_factor);
                int top = int((cy - 0.5f * h) * y_factor);
                int width = int(w * x_factor);
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
    for (int idx : indices)
    {
        detections.push_back({class_ids[idx],
                              confidences[idx],
                              boxes[idx]});
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

bool StereoFusionNode::estimateDepthFromDisparity(
    const std::vector<Detection> &detections,
    const cv::Mat &disparity,
    std::vector<Detection3D> &out_detections_3d)
{
    CV_Assert(disparity.type() == CV_32F);
    CV_Assert(!disparity.empty());

    out_detections_3d.clear();

    const cv::Matx34d P_left  = left_cam_model_.projectionMatrix();
    const cv::Matx34d P_right = right_cam_model_.projectionMatrix();
    double fx = P_left(0,0);
    double cx = P_left(0,2);
    double cy = P_left(1,2);

    double baseline = -(P_right(0,3) - P_left(0,3)) / fx;
    if (baseline <= 0.0 || fx <= 0.0)
    {
        RCLCPP_ERROR(get_logger(), "bad stereo parameters: fx=%.3f baseline=%.3f", fx, baseline);
        return false;
    }

    for (const auto &det : detections)
    {
        cv::Rect roi = det.box & cv::Rect(0, 0, disparity.cols, disparity.rows);
        if (roi.area() < 25)
            continue;

        std::vector<float> disp_vals;
        disp_vals.reserve(roi.area());

        for (int y = roi.y; y < roi.y + roi.height; ++y)
        {
            const float *row = disparity.ptr<float>(y);
            for (int x = roi.x; x < roi.x + roi.width; ++x)
            {
                float d = row[x];
                if (d > 0.1f && std::isfinite(d))
                    disp_vals.push_back(d);
            }
        }
        if (disp_vals.size() < 30)
            continue;

        std::nth_element(
            disp_vals.begin(),
            disp_vals.begin() + disp_vals.size() / 2,
            disp_vals.end());
        float disp = disp_vals[disp_vals.size() / 2];


        double Z = fx * baseline /disp;
        double u = det.box.x + det.box.width*0.5;
        double v = det.box.y + det.box.height*0.5;
        double X = (u-cx)* Z / fx;
        double Y = (v-cy)* Z / fx;

        Detection3D d3;
        d3.det = det;
        d3.disparity = disp;
        d3.position_cam = cv::Point3f(static_cast<float>(X),static_cast<float>(Y), static_cast<float>(Z));
        out_detections_3d.push_back(d3);
    }
    return !out_detections_3d.empty();
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoFusionNode>());
    rclcpp::shutdown();
    return 0;
}
