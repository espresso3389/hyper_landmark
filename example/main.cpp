#include <vector>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "face_landmark_model.h"

int main()
{

    FaceLandmarkModel model("model/haar_facedetection.xml");
    std::string modelFilePath = "model/landmark-model.bin";

    if (!model.loadFaceLandmarkModel(modelFilePath))
    {
        std::cout << "Model Opening Failed." << std::endl;
        std::cin >> modelFilePath;
    }

    cv::VideoCapture mCamera("/home/ub/work/PXL_20240711_101953416.LS.mp4");
    if (!mCamera.isOpened())
    {
        std::cout << "Camera Opening Failed..." << std::endl;
        return 0;
    }
    cv::Mat image;
    std::vector<cv::Mat> currentShapes(MAX_FACE_NUM);
    while (1)
    {
        mCamera >> image;
        if (image.empty())
            break;

        cv::flip(image, image, 0);
        cv::resize(image, image, cv::Size(), 0.5, 0.5);

        std::cout << "track: " << model.track(image, currentShapes) << std::endl;
        cv::Vec3d eav;
        model.estimateHeadPose(currentShapes[0], eav);
        model.drawPose(image, currentShapes[0], 50);

        for (int i = 0; i < MAX_FACE_NUM; i++)
        {
            if (!currentShapes[i].empty())
            {
                int nLandmarks = currentShapes[i].cols / 2;
                for (int j = 0; j < nLandmarks; j++)
                {
                    int x = currentShapes[i].at<float>(j);
                    int y = currentShapes[i].at<float>(j + nLandmarks);

                    cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        cv::imshow("Preview", image);
        if (27 == cv::waitKey(5))
            break;
    }

    mCamera.release();
    cv::destroyAllWindows();
    return 0;
}
