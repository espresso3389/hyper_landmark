#ifndef FEATURE_DESCRIPTOR_H_
#define FEATURE_DESCRIPTOR_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "cereal/cereal.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal_extension/mat_cerealisation.hpp"

extern "C"
{
#include "hog.h" // From the VLFeat C library
}

struct HoGParam
{
    VlHogVariant vlHogVariant;
    int numCells;
    int cellSize;
    int numBins;
    float relativePatchSize; // the patch size we'd like in percent of the IED of the current image
                             // note: alternatively, we could dynamically vary cellSize. Guess it works if the hog features are somehow normalised.

private:
    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template <class Archive>
    void serialize(Archive &ar)
    {
        ar(vlHogVariant, numCells, cellSize, numBins, relativePatchSize);
    }
};

cv::Mat calculateHogDescriptor(cv::Mat &src, cv::Mat shape, const std::vector<int> &landmarkIndices, const std::vector<int> &eyeIndices, HoGParam hogParam);

#endif
