#pragma once

#include <support-common.h>

#ifdef __cplusplus
#include <opencv/cv.h>
#include <vector>
#endif
#include "classification-c.h"

#ifdef __cplusplus
class Caffe_API Classifier {
public:
	Classifier(const char* prototxt_file,
		const char* caffemodel_file,
		float scale_raw = 1,
		const char* mean_file = "",
		int num_means = 0,
		float* means = 0,
		int gpu_id = -1);

	Classifier(const void* prototxt_data,
		int prototxt_data_length,
		const void* caffemodel_data,
		int caffemodel_data_length,
		float scale_raw = 1,
		const char* mean_file = "",
		int num_means = 0,
		float* means = 0,
		int gpu_id = -1);

	virtual ~Classifier();

public:
	SoftmaxResult* predictSoftmax(const cv::Mat& img, int top_n = 5);
	BlobData* extfeature(const cv::Mat& img, const char* layer_name = 0);

	int input_num(int index = 0);
	int input_channels(int index = 0);
	int input_width(int index = 0);
	int input_height(int index = 0);

	BlobData* getOutputBlob(int index);
	int getOutputBlobCount();
	BlobData* getBlobData(const char* blob_name);

private:
	//Blob<float>
	BlobData* getBlobDataByRawBlob(void* blob);
	void SetMean(const char* mean_file);
	void Predict(const cv::Mat& img, std::vector<std::vector<float> >& out);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	float scale_raw;
	//Net<float> net_;
	void* net_;
};


inline void WPtr<BlobData*>::release(BlobData* p){
	releaseBlobData(p);
}

inline void WPtr<SoftmaxResult*>::release(SoftmaxResult* p){
	releaseSoftmaxResult(p);
}
#endif