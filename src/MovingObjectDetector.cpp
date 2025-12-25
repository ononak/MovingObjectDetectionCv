
#include "MovingObjectDetector.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/video/background_segm.hpp>

constexpr int maxContourSize = 200;

MovingObjectDetector::MovingObjectDetector(Method method) {
  if (method == Method::KNN) {
    mBackgroundSubtractor = cv::createBackgroundSubtractorKNN();
  } else {
    mBackgroundSubtractor = cv::createBackgroundSubtractorMOG2();
  }
  mKernel3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
  mKernel7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7});
};

std::vector<cv::Rect> MovingObjectDetector::detect(const cv::Mat &frame) {

  mBackgroundSubtractor->apply(frame, mForeGroundMask);
  postProcessMask();
  return getObjectBoundingBoxes();
}

void MovingObjectDetector::postProcessMask() {
  // Remove shadows if detectShadows=true (shadows are 127)
  cv::threshold(mForeGroundMask, mForeGroundMask, 200, 255, cv::THRESH_BINARY);
  // speckle removal
  cv::morphologyEx(mForeGroundMask, mForeGroundMask, cv::MORPH_OPEN, mKernel3);
  // fill holes / connect parts
  cv::morphologyEx(mForeGroundMask, mForeGroundMask, cv::MORPH_CLOSE, mKernel7);

}

std::vector<cv::Rect> MovingObjectDetector::getObjectBoundingBoxes()
{
      // Find contours of the moving objects
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mForeGroundMask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // Filter out small contours
  contours.erase(std::remove_if(contours.begin(), contours.end(),
                                [](const std::vector<cv::Point> &contour) {
                                  return cv::contourArea(contour) <
                                         maxContourSize;
                                }),
                 contours.end());

  std::vector<cv::Rect> boundingBoxes;
  for (const auto &contour : contours) {
    cv::Rect boundingBox = cv::boundingRect(contour);
    boundingBoxes.push_back(boundingBox);
  }
  return boundingBoxes;
}