#pragma once
#ifndef LDMARKMODEL_H_
#define LDMARKMODEL_H_

#include <iostream>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "cereal/cereal.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal_extension/mat_cerealisation.hpp"

#include "helper.h"
#include "feature_descriptor.h"

#define SDM_NO_ERROR 0         // �޴���
#define SDM_ERROR_FACEDET 200  // ����ͨ��CascadeClassifier��⵽����
#define SDM_ERROR_FACEPOS 201  // ����λ�ñ仯�ϴ󣬿���
#define SDM_ERROR_FACESIZE 202 // ������С�仯�ϴ󣬿���
#define SDM_ERROR_FACENO 203   // �Ҳ�������
#define SDM_ERROR_IMAGE 204    // ͼ�����

#define SDM_ERROR_ARGS 400  // �������ݴ���
#define SDM_ERROR_MODEL 401 // ģ�ͼ��ش���

#define MAX_FACE_NUM 3 // ����ͬʱ������������

// �ع�����
class LinearRegressor
{

public:
    LinearRegressor();

    cv::Mat predict(cv::Mat values);

private:
    cv::Mat weights;
    cv::Mat eigenvectors;
    cv::Mat meanvalue;
    cv::Mat x;
    bool isPCA;

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(weights, meanvalue, x, isPCA);
        if (isPCA)
        {
            ar(eigenvectors);
        }
    }
};

class FaceLandmarkModel
{

public:
    FaceLandmarkModel(std::string faceModel);

    void loadFaceDetModelFile(std::string filePath);

    int track(const cv::Mat &src, std::vector<cv::Mat> &current_shape, bool isDetFace = false);

    void estimateHeadPose(cv::Mat &current_shape, cv::Vec3d &eav);

    void drawPose(cv::Mat &img, const cv::Mat &current_shape, float lineL = 50);

private:
    std::string faceModelPath;

    std::vector<cv::Rect> faceBox;

    std::vector<std::vector<int>> landmarkIndices;
    std::vector<int> eyeIndices;
    cv::Mat meanShape;
    std::vector<HoGParam> HoGParams;
    bool isNormal;
    std::vector<LinearRegressor> linearRegressors;
    cv::CascadeClassifier faceCascade;

    cv::Mat estimateHeadPoseMat;
    cv::Mat estimateHeadPoseMat2;
    int *estimateHeadPosePointIndices;

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(landmarkIndices, eyeIndices, meanShape, HoGParams, isNormal, linearRegressors);
    }
};

// ����ģ��
bool load_face_landmark_model(std::string filename, FaceLandmarkModel &model);

#endif
