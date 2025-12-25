

#include "src/MovingObjectDetector.h"
#include <opencv2/opencv.hpp>
#include <iostream>


using namespace std;
using namespace cv;

int main() {

  std::string videoPath = "../human_1.mp4";

  // Load an video from file
  VideoCapture cap(videoPath);
  if (!cap.isOpened()) {
    cerr << "Error opening video file" << endl;
    return -1;
  }

  MovingObjectDetector detector;

  while (true) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) {
      break; // End of video
    }

    auto boxes = detector.detect(frame);

    for (const auto &box : boxes) {
      cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("Detected moving objects", frame);

    if (cv::waitKey(10) >= 0) {
      break; // Exit on any key press
    }
  }

    return 0;
}