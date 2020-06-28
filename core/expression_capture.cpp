#include "expression_capture.h"

#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

#include <opencv2/opencv.hpp>

#include <torch/all.h>
#include <torch/script.h>

#define SRGB_TO_LINEAR(pf) ((pf<=0.04045f)?(pf/12.92f):std::pow(((pf + 0.055f)/1.055f),2.4f))
#define LINEAR_TO_SRGB(x) ((x<=0.003130804953560372f)?(x*12.92f):(1.055f*(std::pow(x,(1.0f/2.4f)))-0.055f))
#define FLIP_FOR_U8C(x) (x<0.0f?0.0f:(x>255.0f?255.0f:x))

typedef std::vector<uint32_t> M_UINT_Arrary;
typedef std::vector<std::pair<cv::Point2d, cv::Point2d>> m_line;

const static M_UINT_Arrary LEFT_EYE_HORIZ_POINTS {36, 39};
const static M_UINT_Arrary LEFT_EYE_TOP_POINTS {37, 38};
const static M_UINT_Arrary LEFT_EYE_BOTTOM_POINTS {41, 40};

const static M_UINT_Arrary RIGHT_EYE_HORIZ_POINTS {42, 45};
const static M_UINT_Arrary RIGHT_EYE_TOP_POINTS {43, 44};
const static M_UINT_Arrary RIGHT_EYE_BOTTOM_POINTS {47, 46};

const static M_UINT_Arrary MOUTH_TOP_POINTS {61, 62, 63};
const static M_UINT_Arrary MOUTH_BOTTOM_POINTS {67, 66, 65};
const static M_UINT_Arrary MOUTH_HORIZ_POINTS {60, 64};


const std::vector<cv::Point3d> reprojectsrc = {cv::Point3f(10, 10.5, 17.0),
                                               cv::Point3f(7, 7.5, 7.0),
                                               cv::Point3f(7, -7.5, 7.0),
                                               cv::Point3f(10, -10.5, 17.0),
                                               cv::Point3f(-10, 10.5, 17.0),
                                               cv::Point3f(-7.0, 7.5, 7.0),
                                               cv::Point3f(-7.0, -7.5, 7.0),
                                               cv::Point3f(-10.0, -10.5, 17.0)
                                              };
const std::vector<cv::Point3d> object_pts = {cv::Point3f(6.825897f, 6.760612f, 4.402142f),
                                             cv::Point3f(1.330353f, 7.122144f, 6.903745f),
                                             cv::Point3f(-1.330353f, 7.122144f, 6.903745f),
                                             cv::Point3f(-6.825897f, 6.760612f, 4.402142f),
                                             cv::Point3f(5.311432f, 5.485328f, 3.987654f),
                                             cv::Point3f(1.789930f, 5.393625f, 4.413414f),
                                             cv::Point3f(-1.789930f, 5.393625f, 4.413414f),
                                             cv::Point3f(-5.311432f, 5.485328f, 3.987654f),
                                             cv::Point3f(2.005628f, 1.409845f, 6.165652f),
                                             cv::Point3f(-2.005628f, 1.409845f, 6.165652f)
                                            };
static double K[9] = {640, 0.0, 640 / 2, 0.0, 640, 480 / 2, 0.0, 0.0, 1.0};
static double D[5] = { 0, 0, 0.0, 0.0, 0};




static cv::VideoCapture video;
static dlib::frontal_face_detector detector;
static dlib::shape_predictor landmark_locator;

static torch::Tensor current_pose;
static torch::Tensor last_pose;
static torch::Tensor source_tensor;
static torch::Device tdevice(at::kCUDA, 0);
static torch::jit::script::Module module1, module2, module3;

static cv::Mat camera_frame;
static cv::Mat result_frame;

static bool torch_is_ready = false;




static double compute_eye_normalized_ratio(dlib::full_object_detection face_landmarks, M_UINT_Arrary eye_horiz_points, M_UINT_Arrary eye_bottom_points,
                                           M_UINT_Arrary eye_top_points, double min_ratio, double max_ratio)
{
    auto left_eye_horiz_diff = face_landmarks.part(eye_horiz_points[0]) - face_landmarks.part(eye_horiz_points[1]);

    auto left_eye_horiz_length = std::sqrt(std::pow(left_eye_horiz_diff.x(), 2) + std::pow(left_eye_horiz_diff.y(), 2));

    auto left_eye_top_point = (face_landmarks.part(eye_top_points[0]) + face_landmarks.part(eye_top_points[1])) / 2.0;
    auto left_eye_bottom_point = (face_landmarks.part(eye_bottom_points[0]) + face_landmarks.part(eye_bottom_points[1])) / 2.0;
    auto left_eye_vert_diff = left_eye_top_point - left_eye_bottom_point;
    auto left_eye_vert_length = std::sqrt(std::pow(left_eye_vert_diff.x(), 2) + std::pow(left_eye_vert_diff.y(), 2));
    auto left_eye_ratio = left_eye_vert_length / left_eye_horiz_length;
    auto left_eye_normalized_ratio = (std::min(std::max(left_eye_ratio, min_ratio), max_ratio) - min_ratio) / (max_ratio - min_ratio);
    return left_eye_normalized_ratio;
}

static double compute_left_eye_normalized_ratio(dlib::full_object_detection face_landmarks, double min_ratio, double max_ratio)
{
    return compute_eye_normalized_ratio(face_landmarks, LEFT_EYE_HORIZ_POINTS, LEFT_EYE_BOTTOM_POINTS, LEFT_EYE_TOP_POINTS, min_ratio, max_ratio);
}


static double compute_right_eye_normalized_ratio(dlib::full_object_detection face_landmarks, double min_ratio, double max_ratio)
{
    return compute_eye_normalized_ratio(face_landmarks, RIGHT_EYE_HORIZ_POINTS, RIGHT_EYE_BOTTOM_POINTS, RIGHT_EYE_TOP_POINTS, min_ratio, max_ratio);
}

static double compute_mouth_normalized_ratio(dlib::full_object_detection face_landmarks, double min_mouth_ratio, double max_mouth_ratio)
{
    auto mouth_top_point = (face_landmarks.part(MOUTH_TOP_POINTS[0])
                            + face_landmarks.part(MOUTH_TOP_POINTS[1])
                            + face_landmarks.part(MOUTH_TOP_POINTS[2])) / 3.0;
    auto mouth_bottom_point = (face_landmarks.part(MOUTH_BOTTOM_POINTS[0])
                               + face_landmarks.part(MOUTH_BOTTOM_POINTS[1])
                               + face_landmarks.part(MOUTH_BOTTOM_POINTS[2])) / 3.0;
    auto mouth_vert_diff = mouth_top_point - mouth_bottom_point;
    auto mouth_vert_length = std::sqrt(std::pow(mouth_vert_diff.x(), 2) + std::pow(mouth_vert_diff.y(), 2));
    auto mouth_horiz_diff = face_landmarks.part(MOUTH_HORIZ_POINTS[0]) - face_landmarks.part(MOUTH_HORIZ_POINTS[1]);
    auto mouth_horiz_length = std::sqrt(std::pow(mouth_horiz_diff.x(), 2) + std::pow(mouth_horiz_diff.y(), 2));
    auto mouth_ratio = mouth_vert_length / mouth_horiz_length;

    auto mouth_normalized_ratio = (std::min(std::max(mouth_ratio, min_mouth_ratio), max_mouth_ratio) - min_mouth_ratio) / (
                                          max_mouth_ratio - min_mouth_ratio);
    return mouth_normalized_ratio;
}

static bool solve_head_pose(const dlib::full_object_detection &face_landmarks, double *euler_angles, m_line &face_box_points)
{
    M_UINT_Arrary indices = {17, 21, 22, 26, 36, 39, 42, 45, 31, 35};

    std::vector<cv::Point2d> image_pts;

    for (const auto &k : indices) {
        auto part = face_landmarks.part(k);
        image_pts.push_back(cv::Point2d(part.x(), part.y()));
    }

    auto cam_matrix = cv::Mat(3, 3, CV_64FC1, K);
    auto dist_coeffs = cv::Mat(5, 1, CV_64FC1, D);

    cv::Mat rotation_vec;
    cv::Mat rotation_mat;
    cv::Mat translation_vec;
    auto pose_mat = cv::Mat(3, 4, CV_64FC1);
    auto euler_angle = cv::Mat(3, 1, CV_64FC1);

    std::vector<cv::Point2d> reprojectdst;
    reprojectdst.resize(8);

    auto out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    auto out_rotation = cv::Mat(3, 3, CV_64FC1);
    auto out_translation = cv::Mat(3, 1, CV_64FC1);

    if (cv::solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, rotation_vec, translation_vec)) {

        cv::projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs, reprojectdst);

        cv::Rodrigues(rotation_vec, rotation_mat);
        cv::hconcat(rotation_mat, translation_vec, pose_mat);
        cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);


        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[0], reprojectdst[1]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[1], reprojectdst[2]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[2], reprojectdst[3]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[3], reprojectdst[0]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[4], reprojectdst[5]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[5], reprojectdst[6]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[6], reprojectdst[7]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[7], reprojectdst[4]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[0], reprojectdst[4]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[1], reprojectdst[5]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[2], reprojectdst[6]));
        face_box_points.push_back(std::pair<cv::Point2d, cv::Point2d>(reprojectdst[3], reprojectdst[7]));

        euler_angles[0] = euler_angle.at<double>(0);
        euler_angles[1] = euler_angle.at<double>(1);
        euler_angles[2] = euler_angle.at<double>(2);

        return true;
    }
    return false;
}

static void draw_face_landmarks(cv::Mat &frame, const dlib::full_object_detection &face_landmarks)
{
    for (uint32_t k = 0; k < 68; ++k) {
        auto part = face_landmarks.part(k);
        cv::rectangle(frame, cv::Point(part.x() - 1, part.y() - 1), cv::Point(part.x() + 1, part.y() + 1), cv::Scalar(0, 255, 0), 1);
    }
}

static void draw_face_box(cv::Mat &frame, const m_line &face_box_points)
{
    for (auto const &k : face_box_points) {
        cv::line(frame, k.first, k.second, cv::Scalar(0, 255, 0), 1);
    }
}

torch::Tensor mat_to_tensor(const cv::Mat &input)
{
    cv::Mat image_transfomed;
    cv::resize(input, image_transfomed, cv::Size(input.cols, input.cols));
    if (input.channels() < 4) {
        cv::cvtColor(input, image_transfomed, cv::COLOR_BGR2BGRA);
    }
    static cv::Mat M(input.rows, input.rows, CV_32FC4, cv::Scalar::all(0));

    if (input.rows != M.rows || input.cols != M.cols) {
        M.create(input.rows, input.rows, CV_32FC4);
    }

    int rows = image_transfomed.rows;
    int cols = image_transfomed.cols;

    for (int i = 0; i < rows ; i++) {
        for (int j = 0; j < cols ; j++) {
            M.at<cv::Vec4f>(i, j)[0] = SRGB_TO_LINEAR(image_transfomed.at<cv::Vec4b>(i, j)[0] / 255.0f);
            M.at<cv::Vec4f>(i, j)[1] = SRGB_TO_LINEAR(image_transfomed.at<cv::Vec4b>(i, j)[1] / 255.0f);
            M.at<cv::Vec4f>(i, j)[2] = SRGB_TO_LINEAR(image_transfomed.at<cv::Vec4b>(i, j)[2] / 255.0f);
            M.at<cv::Vec4f>(i, j)[3] = image_transfomed.at<cv::Vec4b>(i, j)[3] / 255.0f;
            //std::cout<<M.at<cv::Vec4d>(i,j)[0]<<M.at<cv::Vec4d>(i,j)[1]<<M.at<cv::Vec4d>(i,j)[2]<<M.at<cv::Vec4d>(i,j)[3]<<std::endl;
        }
    }
    //std::cout<<input<<std::endl;

    M = M.reshape(1, rows * rows);
    M = M.t();
    M = M.reshape(rows, 4);

    return torch::from_blob(M.data, {4, M.cols, M.cols}, torch::kFloat32) * 2.0f - 1.0f;
}

static void tensor_to_mat(const torch::Tensor &tensor_image, cv::Mat &output)
{
    auto height = tensor_image.size(1);
    auto width = tensor_image.size(2);

    auto torch_image = (tensor_image + 1.0) * 0.5;

    static cv::Mat resultImg(1, width * height * 4, CV_32FC1);

    if (width * height != resultImg.cols) {
        resultImg.create(1, width * height * 4, CV_32FC1);
    }


    std::memcpy((void *) resultImg.data, torch_image.data_ptr(), sizeof(float) * torch_image.numel());
    resultImg = resultImg.reshape(1, 4);
    resultImg = resultImg.t();
    resultImg = resultImg.reshape(4, height);

    int rows = resultImg.rows;
    int cols = resultImg.cols;

    for (int i = 0; i < rows ; i++) {
        for (int j = 0; j < cols ; j++) {
            resultImg.at<cv::Vec4f>(i, j)[0] = FLIP_FOR_U8C(LINEAR_TO_SRGB(resultImg.at<cv::Vec4f>(i, j)[0]) * 255);
            resultImg.at<cv::Vec4f>(i, j)[1] = FLIP_FOR_U8C(LINEAR_TO_SRGB(resultImg.at<cv::Vec4f>(i, j)[1]) * 255);
            resultImg.at<cv::Vec4f>(i, j)[2] = FLIP_FOR_U8C(LINEAR_TO_SRGB(resultImg.at<cv::Vec4f>(i, j)[2]) * 255);
            resultImg.at<cv::Vec4f>(i, j)[3] = FLIP_FOR_U8C(resultImg.at<cv::Vec4f>(i, j)[3] * 255);
        }
    }
    resultImg.convertTo(output, CV_8UC4);
}




namespace  expc
{

bool init()
{
    if (video.isOpened()) {
        video.release();
    }

    detector = dlib::get_frontal_face_detector();
    dlib::deserialize("data/shape_predictor_68_face_landmarks.dat") >> landmark_locator;

    torch_is_ready = torch::cuda::is_available();
    if (!torch_is_ready) {
        return false;
    }

    try {
        module1 = torch::jit::load("data/module1.pt");
        module2 = torch::jit::load("data/module2.pt");
        module3 = torch::jit::load("data/module3.pt");

        module1.to(tdevice);
        module2.to(tdevice);
        module3.to(tdevice);
    } catch (const c10::Error &e) {
        std::cout << e.msg() << std::endl;
        torch_is_ready = false;
        return false;
    }
    return  true;
}

void uninit()
{
    video.release();
}

bool open_camera(int index)
{
    cv::Mat temp;
    if (video.open(index)) {
        video >> temp;

        K[0] = temp.cols;
        K[2] = temp.cols / 2.0;
        K[4] = temp.cols;
        K[5] = temp.rows / 2.0;
        return true;
    }
    return false;
}

void close_camera()
{
    video.release();
}

void set_source_image(const std::string &filepath)
{
    auto source_image = cv::imread(filepath, cv::IMREAD_UNCHANGED);
    result_frame.create(source_image.rows, source_image.cols, CV_8UC4);
    source_tensor = mat_to_tensor(source_image).to(tdevice).unsqueeze(0);
}

M_Frame get_image()
{
    M_Frame frame;
    try {
        if (!video.isOpened())
            return frame;

        video >> camera_frame;
        cv::cvtColor(camera_frame, camera_frame, cv::COLOR_BGR2RGB);
        dlib::cv_image<dlib::rgb_pixel> cimg(camera_frame);
        auto dets = detector(cimg);

        if (dets.size() > 0) {
            dlib::full_object_detection face_landmarks = landmark_locator(cimg, dets[0]);

            m_line face_box_points;
            double euler_angles[3];
            if (solve_head_pose(face_landmarks, euler_angles, face_box_points)) {
                draw_face_box(camera_frame, face_box_points);
                draw_face_landmarks(camera_frame, face_landmarks);
                cv::flip(camera_frame, camera_frame, 1);
            }
        }

        frame.height = camera_frame.rows;
        frame.width = camera_frame.cols;
        frame.data = camera_frame.data;
        frame.type = camera_frame.type();

    } catch (cv::Exception &e) {
        std::cout << e.msg << std::endl;
    }
    return frame;
}

M_Frames get_image_predicted()
{
    M_Frame frame1, frame2;

    try {
        if (!video.isOpened())
            return M_Frames(frame1, frame2);

        video >> camera_frame;

        if (camera_frame.empty())
            return M_Frames(frame1, frame2);

        //if(!source_tensor.is_same(torch::Tensor()))
        //   cv::resize(camera_frame,camera_frame,cv::Size(source_tensor.size(2),source_tensor.size(3)),0,0,cv::INTER_LINEAR);

        cv::cvtColor(camera_frame, camera_frame, cv::COLOR_BGR2RGB);
        dlib::cv_image<dlib::rgb_pixel> cimg(camera_frame);
        std::vector<dlib::rectangle> dets = detector(cimg);

        if (dets.size() > 0) {
            auto face_landmarks = landmark_locator(cimg, dets[0]);

            m_line face_box_points;
            double euler_angles[3];
            if (solve_head_pose(face_landmarks, euler_angles, face_box_points)) {

                draw_face_box(camera_frame, face_box_points);
                draw_face_landmarks(camera_frame, face_landmarks);

                if (torch_is_ready) {
                    if (!source_tensor.is_same(torch::Tensor())) {

                        current_pose = torch::zeros(6, tdevice);
                        current_pose[0] = std::max(std::min(-euler_angles[0] / 15.0, 1.0), -1.0);
                        current_pose[1] = std::max(std::min(-euler_angles[1] / 15.0, 1.0), -1.0);
                        current_pose[2] = std::max(std::min(euler_angles[2] / 15.0, 1.0), -1.0);

                        if (last_pose.is_same(torch::Tensor())) {
                            last_pose = current_pose;
                        } else {
                            current_pose = current_pose * 0.5 + last_pose * 0.5;
                            last_pose = current_pose;
                        }

                        auto eye_min_ratio = 0.15;
                        auto eye_max_ratio = 0.20;
                        auto left_eye_normalized_ratio = compute_left_eye_normalized_ratio(face_landmarks, eye_min_ratio, eye_max_ratio);
                        current_pose[3] = 1 - left_eye_normalized_ratio;
                        auto right_eye_normalized_ratio = compute_right_eye_normalized_ratio(face_landmarks, eye_min_ratio, eye_max_ratio);

                        current_pose[4] = 1 - right_eye_normalized_ratio;

                        auto min_mouth_ratio = 0.02;
                        auto max_mouth_ratio = 0.3;
                        auto mouth_normalized_ratio = compute_mouth_normalized_ratio(face_landmarks, min_mouth_ratio, max_mouth_ratio);

                        current_pose[5] = mouth_normalized_ratio;
                        current_pose = current_pose.unsqueeze(0);

                        torch::Tensor rotate_params, morph_params;

                        morph_params = current_pose.index_select(1, torch::range(3, 5, 1, torch::kLong).to(current_pose.device()));
                        rotate_params = current_pose.index_select(1, torch::range(0, 2, 1, torch::kLong).to(current_pose.device()));


                        try {
                            std::vector<torch::jit::IValue> inputs;
                            inputs.push_back(source_tensor);
                            inputs.push_back(morph_params);

                            auto output1 = module1.forward(inputs).toList();
                            inputs.clear();
                            inputs.push_back(output1[0]);
                            inputs.push_back(rotate_params);
                            auto output2 = module2.forward(inputs).toList();

                            inputs.clear();
                            inputs.push_back(output2[0]);
                            inputs.push_back(output2[1]);
                            inputs.push_back(rotate_params);

                            auto output3 = module3.forward(inputs).toTensorVector();

                            tensor_to_mat(output3[0][0].detach().cpu(), result_frame);

                            frame2.height = result_frame.rows;
                            frame2.width = result_frame.cols;
                            frame2.data = result_frame.data;
                            frame2.type = result_frame.type();
                        } catch (const c10::Error &e) {
                            std::cout << e.msg() << std::endl;
                        }
                    }
                }
            }
        }

        cv::flip(camera_frame, camera_frame, 1);

        frame1.height = camera_frame.rows;
        frame1.width = camera_frame.cols;
        frame1.data = camera_frame.data;
        frame1.type = camera_frame.type();
    } catch (cv::Exception &e) {
        std::cout << e.msg << std::endl;
    }

    return M_Frames(frame1, frame2);
}
}
