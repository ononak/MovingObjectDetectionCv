#ifndef MOVINGOBJECTDETECTOR_H
#define MOVINGOBJECTDETECTOR_H

#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

class MovingObjectDetector {

public:
  enum class Method { KNN, GAUSSIAN };

private:
  cv::Mat mBackgroundFrame;
  cv::Mat mForeGroundMask;
  cv::Mat mKernel3; // small kernel for noise removal
  cv::Mat mKernel7; // larger kernel for hole filling
  cv::Ptr<cv::BackgroundSubtractor> mBackgroundSubtractor;

public:
  explicit MovingObjectDetector(Method method = Method::GAUSSIAN);
  void update(const cv::Mat &frame);
  std::vector<cv::Rect> detect(const cv::Mat &frame);
  cv::Mat getBackground() const { return mBackgroundFrame; };
  cv::Mat getForegroundMask() const { return mForeGroundMask; };

  private:
    void postProcessMask();
    std::vector<cv::Rect> getObjectBoundingBoxes();
};

#endif // MOVINGOBJECTDETECTOR_H