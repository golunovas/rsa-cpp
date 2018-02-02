#include <iostream>

#include <opencv2/opencv.hpp>
#include "rsa_face_detector.hpp"
#include "helpers.hpp"

int main() {
	Timer timer;

	RSAFaceDetector rsa("./model", 2048, false, 0);
	cv::Mat img = cv::imread("test.jpg");

	timer.start();
	auto faces = rsa.detect(img);
	std::cout << "Estimated time: " << timer.stop() << std::endl;
	
	for (auto& face : faces) {
		std::cout << face.score << std::endl;
		cv::rectangle(img, face.rect, cv::Scalar(255, 0, 0));
		for (auto& pt : face.pts) {
			cv::circle(img, pt, 3, cv::Scalar(0, 0, 255));
		}
	}
	cv::imwrite("output.jpg", img);
	return 0;
}