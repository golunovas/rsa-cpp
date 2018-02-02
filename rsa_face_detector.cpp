#include "rsa_face_detector.hpp"
#include "similarity_transform.hpp"

const std::string NET1_PROTO = "/res_pool2.prototxt";
const std::string NET1_WEIGHTS = "/resnet50.caffemodel";

const std::string NET2_PROTO = "/hm_trans.prototxt";
const std::string NET2_WEIGHTS = "/hm_trans.caffemodel";

const std::string NET3_PROTO = "/res_3b_s16_f2r.prototxt";
const std::string NET3_WEIGHTS = "/resnet50.caffemodel";

const std::vector<float> SCALE_POWERS = { 5.f, 4.f, 3.f, 2.f, 1.f};

const float ANCHOR_RECT_SIDE = 44.7548f - (-44.7548f);
const float ANCHOR_CENTER = 7.5f;

const std::vector<cv::Point2f> ANCHOR_PTS = { 
	{-0.1719f, -0.2204f}, 
	{0.1719f, -0.2261f}, 
	{-0.0017f, 0.0047f}, 
	{-0.1409f, 0.2034f}, 
	{0.1409f, 0.1978f} 
};

const int FINAL_STRIDE = 16;
const float CLS_THRESHOLD = 3.f;
const float NMS_RATIO = 0.2f;
const float NMS_THRESHOLD = 8.f;

RSAFaceDetector::RSAFaceDetector(const std::string& modelDir, int maxImageSide, bool useGPU, int deviceID) : 
	maxImageSide_(maxImageSide) {
	if (useGPU) {
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
		caffe::Caffe::SetDevice(deviceID);
	} else {
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
	}
	net1_.reset( new caffe::Net<float> (modelDir + NET1_PROTO, caffe::TEST));
	net1_->CopyTrainedLayersFrom(modelDir + NET1_WEIGHTS);
	net2_.reset( new caffe::Net<float> (modelDir + NET2_PROTO, caffe::TEST));
	net2_->CopyTrainedLayersFrom(modelDir + NET2_WEIGHTS);
	net3_.reset( new caffe::Net<float> (modelDir + NET3_PROTO, caffe::TEST));
	net3_->CopyTrainedLayersFrom(modelDir + NET3_WEIGHTS);
}

std::vector<RSAFace> RSAFaceDetector::detect(cv::Mat img) {
	float resizeFactor = maxImageSide_ / static_cast<float>(std::max(img.cols, img.rows));
	cv::Mat resizedImg;
	cv::resize(img, resizedImg, cv::Size(std::round(img.cols * resizeFactor), std::round(img.rows * resizeFactor)));
	resizedImg.convertTo(resizedImg, CV_32FC3);
	setNetInput(net1_, resizedImg);
	net1_->Forward();
	std::vector<caffe::Blob<float> > featureMapBlobs(SCALE_POWERS.size());
	featureMapBlobs[0].CopyFrom(*(net1_->blob_by_name("res2b")), false, true);
	std::vector<float> scales = SCALE_POWERS;
	for (size_t i = 1; i < scales.size(); ++i) {
		int itersCount = std::round(scales[i - 1] - scales[i]);
		net2_->input_blobs()[0]->CopyFrom(featureMapBlobs[i - 1], false, true);
		net2_->Reshape();
		for (int j = 0; j < itersCount - 1; ++j) {
			net2_->Forward();
			net2_->input_blobs()[0]->CopyFrom(*(net2_->blob_by_name("res2b_trans_5")), false, true);
			net2_->Reshape();
		}
		net2_->Forward();
		featureMapBlobs[i].CopyFrom(*(net2_->blob_by_name("res2b_trans_5")), false, true);
	}
	std::for_each(scales.begin(), scales.end(), [](float &x) { x = std::pow(2.f, x - 5.f); } );
	std::vector<RSAFace> faces;
	for (size_t i = 0; i < featureMapBlobs.size(); ++i) {
		net3_->input_blobs()[0]->CopyFrom(featureMapBlobs[i], false, true);
		net3_->Reshape();
		net3_->Forward();
		const caffe::Blob<float>* rpnRegBlob = net3_->blob_by_name("rpn_reg").get();
		const caffe::Blob<float>* rpnClsBlob = net3_->blob_by_name("rpn_cls").get();
		int clsHeight = rpnClsBlob->shape()[2];
		int clsWidth = rpnClsBlob->shape()[3];
		for (int y = 0; y < clsHeight; ++y) {
			for (int x = 0; x < clsWidth; ++x) {
				float score = rpnClsBlob->data_at(0, 0, y, x);
				if (score < CLS_THRESHOLD) {
					continue;
				}
				float anchorCurrentCenterX = x * FINAL_STRIDE + ANCHOR_CENTER;
				float anchorCurrentCenterY = y * FINAL_STRIDE + ANCHOR_CENTER;
				RSAFace currentFace;
				currentFace.pts.resize(ANCHOR_PTS.size());
				for (int p = 0; p < currentFace.pts.size(); ++p) {
					float ptsDeltaX = rpnRegBlob->data_at(0, 2 * p, y, x);
					float ptsDeltaY = rpnRegBlob->data_at(0, 2 * p + 1, y, x);
					currentFace.pts[p].x = ((ANCHOR_PTS[p].x + ptsDeltaX) * ANCHOR_RECT_SIDE + anchorCurrentCenterX)
						/ scales[i] / resizeFactor;
					currentFace.pts[p].y = ((ANCHOR_PTS[p].y + ptsDeltaY) * ANCHOR_RECT_SIDE + anchorCurrentCenterY)
						/ scales[i] / resizeFactor;
				}
				currentFace.rect = ptsToRect(currentFace.pts);
				currentFace.score = score;
				faces.push_back(currentFace);
			}
		}
	}
	faces = nonMaximumSuppression(faces);
	faces.erase(std::remove_if(faces.begin(), faces.end(), [] (const RSAFace& f) { return f.score < NMS_THRESHOLD; }), faces.end());
	return faces;

}

void RSAFaceDetector::setNetInput(boost::shared_ptr< caffe::Net<float> > net, cv::Mat img) {
	std::vector<cv::Mat> channels;
	cv::split(img, channels);   
	caffe::Blob<float>* inputLayer = net->input_blobs()[0];
	assert(inputLayer->channels() == channels.size());
	if (img.rows != inputLayer->height() || img.cols != inputLayer->width()) {
		inputLayer->Reshape(1, channels.size(), img.rows, img.cols);
		net->Reshape();
	}
	float* inputData = inputLayer->mutable_cpu_data();
	for (size_t i = 0; i < channels.size(); ++i) {
		channels[i] -= 127.f;
		memcpy(inputData, channels[i].data, sizeof(float) * img.cols * img.rows);
		inputData += img.cols * img.rows;
	}
}

cv::Rect RSAFaceDetector::ptsToRect(const std::vector<cv::Point2f>& pts) {
	const std::vector<cv::Point2f> originalPts = { {0.2f, 0.2f}, {0.8f, 0.2f}, {0.5f, 0.5f}, {0.3f, 0.75f}, {0.7f, 0.75f} };
	const std::vector<cv::Point2f> originalRectPts = { {0.5f, 0.5f}, {0.0f, 0.0f}, {1.0f, 0.0f}};
	cv::Mat transformMat = calcSimilarityTransform(originalPts, pts);
	std::vector<cv::Point2f> rectPts;
	cv::transform(originalRectPts, rectPts, transformMat);
	float rectSide = std::sqrt(std::pow(rectPts[2].x - rectPts[1].x, 2.f) + std::pow(rectPts[2].y - rectPts[1].y, 2.f));
	cv::Rect r(rectPts[0].x - rectSide / 2.f, rectPts[0].y - rectSide / 2.f, rectSide, rectSide);
	return r;
}

std::vector<RSAFace> RSAFaceDetector::nonMaximumSuppression(std::vector<RSAFace> faces) {
	std::vector<RSAFace> facesNMS;
	if (faces.empty()) {
		return facesNMS;
	}
	std::sort(faces.begin(), faces.end(), [](const RSAFace& f1, const RSAFace& f2) {
		return f1.score > f2.score;
	});
	std::vector<int> indices(faces.size());
	for (size_t i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}
	while (indices.size() > 0) {
		int idx = indices[0];
		facesNMS.push_back(faces[idx]);
		std::vector<int> tmpIndices = indices;
		indices.clear();
		for(size_t i = 1; i < tmpIndices.size(); ++i) {
			int tmpIdx = tmpIndices[i];
			float interX1 = std::max(faces[idx].rect.x, faces[tmpIdx].rect.x);
			float interY1 = std::max(faces[idx].rect.y, faces[tmpIdx].rect.y);
			float interX2 = std::min(faces[idx].rect.x + faces[idx].rect.width, faces[tmpIdx].rect.x + faces[tmpIdx].rect.width);
			float interY2 = std::min(faces[idx].rect.y + faces[idx].rect.height, faces[tmpIdx].rect.y + faces[tmpIdx].rect.height);
			float interWidth = std::max(0.f, (interX2 - interX1 + 1));
			float interHeight = std::max(0.f, (interY2 - interY1 + 1));
			float interArea = interWidth * interHeight;
			float area1 = (faces[idx].rect.width + 1) * (faces[idx].rect.height + 1);
			float area2 = (faces[tmpIdx].rect.width + 1) * (faces[tmpIdx].rect.height + 1);
			float o = 0.f;
			o = interArea / (area1 + area2 - interArea);
			if(o <= NMS_RATIO) {
				indices.push_back(tmpIdx);
			}
		}
	}
	return facesNMS;
}