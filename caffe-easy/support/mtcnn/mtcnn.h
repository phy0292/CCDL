// This file is the main function of CascadeCNN.
// A C++ re-implementation for the paper 
// Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li. Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks. IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499-1503, 2016. 
//
// Code exploited by Feng Wang <feng.wff@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD lisence.
//
// Please cite Zhang's paper in your publications if this code helps your research.
#pragma once
#include <support-common.h>
#include <opencv2\opencv.hpp>
#include <string>
typedef cv::Rect_<double> Rect2d;
using std::string;
using std::vector;
using std::pair;
using cv::Mat;
using cv::Point2d;
using cv::Rect;

class Caffe_API MTCNN {
public:
	struct FaceInfo{
		cv::Rect bbox;
		float score;
		int pointCount;
		Point2d* list;
	};

	struct FaceList{
		int count;
		FaceInfo* list;
	};

	MTCNN(string net12_definition, string net12_weights,
		string net12_stitch_definition, string net12_stitch_weights,
		string net24_definition, string net24_weights,
		string net48_definition, string net48_weights,
		string netLoc_definition, string netLoc_weights,
		int gpu_id = -1);


	MTCNN(const vector<char>& package, int gpu_id = -1);

	FaceList* detect3(const Mat& input_image, double start_scale = 1, double min_confidence = 0.995,
		bool do_nms = true, double nms_threshold = 0.7,
		bool output_points = false);

	vector<pair<Rect, float>> detect2(const Mat& input_image, double start_scale = 1, double min_confidence = 0.995,
		bool do_nms = true, double nms_threshold = 0.7,
		bool output_points = false, vector<vector<Point2d>>& points = vector<vector<Point2d>>());

	vector<Rect> detect(const Mat& input_image, int min_face = 80, double min_confidence = 0.7,
		bool do_nms = true, double nms_threshold = 0.7,
		bool output_points = false, vector<vector<Point2d>>& points = vector<vector<Point2d>>());

private:
	void* _native;
};

void Caffe_API releaseFaceList(MTCNN::FaceList* flist);

inline void WPtr<MTCNN::FaceList>::release(MTCNN::FaceList* p){
	releaseFaceList(p);
}