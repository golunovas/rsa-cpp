#ifndef RSA_FACE_DETECTOR_HPP_
#define RSA_FACE_DETECTOR_HPP_

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

const int NUM_PTS = 5;

struct RSAFace {
	cv::Rect rect;
	std::vector<cv::Point2f> pts;
	float score;
};

class RSAFaceDetector {
private:
	int maxImageSide_;
	boost::shared_ptr< caffe::Net<float> > net1_;
	boost::shared_ptr< caffe::Net<float> > net2_;
	boost::shared_ptr< caffe::Net<float> > net3_;

	void setNetInput(boost::shared_ptr< caffe::Net<float> > net, cv::Mat img);
	cv::Rect ptsToRect(const std::vector<cv::Point2f>& pts);
	std::vector<RSAFace> nonMaximumSuppression(std::vector<RSAFace> faces);
public:
	RSAFaceDetector(const std::string& modelDir, int maxImageSide, bool useGPU = false, int deviceID = 0);
	std::vector<RSAFace> detect(cv::Mat img);
};

#endif // RSA_FACE_DETECTOR_HPP_