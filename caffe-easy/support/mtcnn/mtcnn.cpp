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
#include <fstream>
#include <thread>
#include <future>
#include <opencv2\opencv.hpp>
#include <network.h>
#include "mtcnn.h"
#include "BoundingBox.inc.h"
#include <map>
#undef assert
#define assert(_Expression) if(!((_Expression)))printf("error: %s %d : %s\n", __FILE__, __LINE__, (#_Expression))
#define USE_GPU_MAT 0

std::shared_ptr<caffe::Network> kCaffeBinding = std::make_shared<caffe::Network>();

using namespace cv;
using namespace std;

const int kHeightStart = 640;
const int kWidthStart = 480;

const int kMaxNet12Num = 20;

static map<string, vector<char>> prasePackage(const vector<char>& data){
	map<string, vector<char>> out;
	const char* d = &data[0];
	int len = 0;
	while (d < &data[0] + data.size()){
		memcpy(&len, d, sizeof(len)); d += sizeof(len);
		string name(d, d + len);  d += len;

		memcpy(&len, d, sizeof(len)); d += sizeof(len);
		vector<char> buf(len);
		memcpy(&buf[0], d, len); d += len;
		out[name] = buf;
	}
	return out;
}

class Caffe_API CascadeCNN {
public:
	CascadeCNN() : scale_decay_(0.707) {}

	CascadeCNN(const vector<char>& package, int gpu_id = -1) :scale_decay_(0.707) {
		map<string, vector<char>> mp = prasePackage(package);
		net12 = kCaffeBinding->AddNet(&mp["det1.proto"][0], mp["det1.proto"].size(), &mp["det1.model"][0], mp["det1.model"].size(), gpu_id);
		net12_stitch = kCaffeBinding->AddNet(&mp["det1-stitch.proto"][0], mp["det1-stitch.proto"].size(), &mp["det1.model"][0], mp["det1.model"].size(), gpu_id);
		net24 = kCaffeBinding->AddNet(&mp["det2.proto"][0], mp["det2.proto"].size(), &mp["det2.model"][0], mp["det2.model"].size(), gpu_id);
		net48 = kCaffeBinding->AddNet(&mp["det3.proto"][0], mp["det3.proto"].size(), &mp["det3.model"][0], mp["det3.model"].size(), gpu_id);
		netLoc = kCaffeBinding->AddNet(&mp["det4.proto"][0], mp["det4.proto"].size(), &mp["det4.model"][0], mp["det4.model"].size(), gpu_id);
	}

	CascadeCNN(string net12_definition, string net12_weights,
		string net12_stitch_definition, string net12_stitch_weights,
		string net24_definition, string net24_weights,
		string net48_definition, string net48_weights,
		string netLoc_definition, string netLoc_weights,
		int gpu_id = -1) :
		scale_decay_(0.707) {
		net12 = kCaffeBinding->AddNet(net12_definition, net12_weights, gpu_id);
		net12_stitch = kCaffeBinding->AddNet(net12_stitch_definition, net12_stitch_weights, gpu_id);
		net24 = kCaffeBinding->AddNet(net24_definition, net24_weights, gpu_id);
		net48 = kCaffeBinding->AddNet(net48_definition, net48_weights, gpu_id);
		netLoc = kCaffeBinding->AddNet(netLoc_definition, netLoc_weights, gpu_id);
	}

	//Only work for small images.
	vector<pair<Rect2d, float>> getNet12ProposalAcc(Mat& input_image, double min_confidence = 0.6, double start_scale = 1,
		bool do_nms = true, double nms_threshold = 0.3) {

		int short_side = min(input_image.cols, input_image.rows);
		assert(log(12.0 / start_scale / (double)short_side) / log(scale_decay_) < kMaxNet12Num);
		std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::system_clock::now();
		Mat start_image = input_image;
		if (start_scale != 1) resize(start_image, start_image, Size(0, 0), start_scale, start_scale);
		vector<pair<Rect2d, float>> accumulate_rects;

		//This number(20000) depends on how much GPU memory you have. Please test and modify it to get maximal speed.
		while (start_image.rows * start_image.cols > 20000 && start_image.rows > 12 && start_image.cols > 12) {
			auto net12output = kCaffeBinding->Forward({ start_image }, net12);
			auto bbox = net12output["bounding_box"];
			if (!(net12output["bounding_box"].size[1] == 1 && net12output["bounding_box"].data[0] == 0)) {
				vector<pair<Rect2d, float>> before_nms;
				for (int i = 0; i < net12output["bounding_box"].size[1]; i++) {
					Rect2d this_rect = Rect2d(net12output["bounding_box"].data[i * 5 + 1] / start_scale, net12output["bounding_box"].data[i * 5] / start_scale,
						net12output["bounding_box"].data[i * 5 + 3] / start_scale, net12output["bounding_box"].data[i * 5 + 2] / start_scale);
					before_nms.push_back(make_pair(this_rect, net12output["bounding_box"].data[i * 5 + 4]));
				}
				if (do_nms && before_nms.size() > 1) {
					vector<int> picked = nms_max(before_nms, 0.5);
					for (auto p : picked) {
						accumulate_rects.push_back(before_nms[p]);
					}
				}
				else {
					accumulate_rects.insert(accumulate_rects.end(), before_nms.begin(), before_nms.end());
				}
			}
			start_scale *= scale_decay_;
			resize(start_image, start_image, Size(input_image.cols * start_scale, input_image.rows * start_scale));
		}
		if (start_image.rows > 12 && start_image.cols > 12) {
			std::vector<std::pair<Rect, double>> location_and_scale;
			Mat big_image = getPyramidStitchingImage2(start_image, location_and_scale);
			Mat stitch_image_x = Mat::zeros((big_image.rows - 12 + 1) / 2 + 1, (big_image.cols - 12 + 1) / 2 + 1, CV_32FC1);
			Mat stitch_image_y = Mat::zeros((big_image.rows - 12 + 1) / 2 + 1, (big_image.cols - 12 + 1) / 2 + 1, CV_32FC1);
			Mat stitch_image_receptive_field = Mat::zeros((big_image.rows - 12 + 1) / 2 + 1, (big_image.cols - 12 + 1) / 2 + 1, CV_32FC1);
			for (auto& ls : location_and_scale) {
				Rect rectInOutput = Rect(max(ls.first.x / 2 - 1, 0), max(ls.first.y / 2 - 1, 0), (ls.first.width - 12 + 2) / 2, (ls.first.height - 12 + 2) / 2);
				stitch_image_x(rectInOutput) = ls.first.y;
				stitch_image_y(rectInOutput) = ls.first.x;
				stitch_image_receptive_field(rectInOutput) = 12.0 / (ls.second*start_scale);
			}

			Mat stitch_image;
			merge(vector<Mat>{ stitch_image_receptive_field, stitch_image_y, stitch_image_x }, stitch_image);
			kCaffeBinding->SetMemoryDataLayer("stitch_data", { stitch_image }, net12_stitch);
			auto net12output = kCaffeBinding->Forward({ big_image }, net12_stitch);

			assert(net12output.size() > 0);

			//Mat stitch_image_x_trans = Mat::zeros((big_image.cols - 12 + 1) / 2 + 1, (big_image.rows - 12 + 1) / 2 + 1, CV_32FC1);
			//stitch_image_x_trans.data = (uchar *)(net12output["conv4-2"].data + 1 * stitch_image_x_trans.rows * stitch_image_x_trans.cols);
			//stitch_image_x_trans = stitch_image_x_trans.t();
			//imshow("stitch_image_x_trans", stitch_image_x_trans);
			//waitKey(0);

			//cout << "estimated output size:" << stitch_image_x.size() << endl;
			//cout << "real output size:" << net12output["bb_map"].size[2] << " " << net12output["bb_map"].size[3] << endl;

			if (!(net12output["bounding_box"].size[1] == 1 && net12output["bounding_box"].data[0] == 0)) {
				for (int i = 0; i < net12output["bounding_box"].size[1]; i++) {
					Rect2d this_rect = Rect2d(net12output["bounding_box"].data[i * 5 + 1], net12output["bounding_box"].data[i * 5],
						net12output["bounding_box"].data[i * 5 + 3], net12output["bounding_box"].data[i * 5 + 2]);
					float score = net12output["bounding_box"].data[i * 5 + 4];
					if (score > 1) {
						cout << "WARNING: softmax confidence overflow! " << this_rect << " " << score << endl;
						continue;
					}
					accumulate_rects.push_back(make_pair(this_rect, score));
				}
			}
		}

		vector<pair<Rect2d, float>> result;

		if (do_nms) {
			//std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
			vector<int> picked = nms_max(accumulate_rects, nms_threshold);
			for (auto& p : picked) {
				//make_rect_square(rect_for_test[p].first);
				result.push_back(accumulate_rects[p]);

			}
			//nms_avg(rects, scores, nms_threshold);
			//cout << "nms time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - p1).count() / 1000 << "ms" << endl;
		}
		else {
			result = accumulate_rects;
		}

		//for (auto& rect : result) {
		//  if (rect.second > 0.6) {
		//    rectangle(input_image, rect.first, Scalar(0, 255, 0), 1);
		//  }
		//}
		//
		//imshow("proposal", input_image);
		//waitKey(0);

		return result;
	}

	vector<pair<Rect2d, float>> getNet12Proposal(Mat& input_image, double min_confidence = 0.6, double start_scale = 1,
		bool do_nms = true, double nms_threshold = 0.3) {
		int short_side = min(input_image.cols, input_image.rows);
		assert(log(12.0 / start_scale / (double)short_side) / log(scale_decay_) < kMaxNet12Num);
		std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::system_clock::now();
		vector<double> scales;
		double scale = start_scale;
		if (floor(input_image.rows * scale) < 1200 && floor(input_image.cols * scale) < 1200) {
			scales.push_back(scale);
		}
		do {
			scale *= scale_decay_;
			if (floor(input_image.rows * scale) < 1200 && floor(input_image.cols * scale) < 1200) {
				scales.push_back(scale);
			}
		} while (floor(input_image.rows * scale * scale_decay_) >= 12 && floor(input_image.cols * scale * scale_decay_) >= 12);

		vector<vector<pair<Rect2d, float>>> sub_rects(scales.size());
#if USE_GPU_MAT
		cuda::GpuMat gpu_input_image;
		gpu_input_image.upload(input_image);
#endif

		for (int s = 0; s < scales.size(); s++) {
			//std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
			//net12_thread_group.create_thread([&]() {
			Mat small_image;
#if USE_GPU_MAT
			cuda::GpuMat gpu_small_image;
			cuda::resize(gpu_input_image, gpu_small_image, Size(0, 0), scales[s], scales[s]);
			gpu_small_image.download(small_image);
#else
			resize(input_image, small_image, Size(0, 0), scales[s], scales[s]);
#endif
			auto net12output = kCaffeBinding->Forward({ small_image }, net12);
			if (!(net12output["bounding_box"].size[1] == 1 && net12output["bounding_box"].data[0] == 0)) {
				vector<pair<Rect2d, float>> before_nms;
				for (int i = 0; i < net12output["bounding_box"].size[1]; i++) {
					Rect2d this_rect = Rect2d(net12output["bounding_box"].data[i * 5 + 1] / scales[s], net12output["bounding_box"].data[i * 5] / scales[s],
						net12output["bounding_box"].data[i * 5 + 3] / scales[s], net12output["bounding_box"].data[i * 5 + 2] / scales[s]);
					before_nms.push_back(make_pair(this_rect, net12output["bounding_box"].data[i * 5 + 4]));
				}
				if (do_nms && before_nms.size() > 1) {
					vector<int> picked = nms_max(before_nms, 0.5);
					for (auto p : picked) {
						//cout << before_nms[p].first << " " << before_nms[p].second << endl;
						sub_rects[s].push_back(before_nms[p]);
					}
				}
				else {
					sub_rects[s].insert(sub_rects[s].end(), before_nms.begin(), before_nms.end());
				}
			}
			//});
			//cout << "scale:" << scales[s] << " time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - p1).count() / 1000 << "ms" << endl;
		}
		//net12_thread_group.join_all();
		vector<pair<Rect2d, float>> accumulate_rects;
		for (int s = 0; s < scales.size(); s++) {
			//cout << "scale:" << scales[s] << " rects:" << sub_rects[s].size() << endl;
			accumulate_rects.insert(accumulate_rects.end(), sub_rects[s].begin(), sub_rects[s].end());
		}
		vector<pair<Rect2d, float>> result;
		if (do_nms) {
			//std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
			vector<int> picked = nms_max(accumulate_rects, nms_threshold);
			for (auto& p : picked) {
				//make_rect_square(rect_for_test[p].first);
				result.push_back(accumulate_rects[p]);
			}
			//nms_avg(rects, scores, nms_threshold);
			//cout << "nms time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - p1).count() / 1000 << "ms" << endl;
		}
		else {
			result = accumulate_rects;
		}

		//for (auto& rect : result) {
		//  if (rect.second > 0.6) {
		//    rectangle(input_image, rect.first, Scalar(0, 255, 0), 1);
		//  }
		//}
		//
		//imshow("proposal", input_image);
		//waitKey(0);

		return result;
	}

	vector<pair<Rect2d, float>> getNet24Refined(vector<Mat> sub_images, vector<Rect2d> image_boxes, double min_confidence = 0.7,
		bool do_nms = true, double nms_threshold = 0.3,
		int batch_size = 500,
		bool output_points = false, vector<vector<Point2d>>& points = vector<vector<Point2d>>()) {
		int num = sub_images.size();
		if (num == 0) return vector<pair<Rect2d, float>>();
		assert(sub_images[0].cols == 24 && sub_images[0].rows == 24);
		vector<pair<Rect2d, float>> rect_and_scores;
		vector<vector<Point2d> > allPoints;

		int total_iter = ceil((double)num / (double)batch_size);
		for (int i = 0; i < total_iter; i++) {
			int start_pos = i * batch_size;
			if (i == total_iter - 1) batch_size = num - (total_iter - 1) * batch_size;
			vector<Mat> net_input = vector<Mat>(sub_images.begin() + start_pos, sub_images.begin() + start_pos + batch_size);
			auto net24output = kCaffeBinding->Forward(net_input, net24);
			for (int j = 0; j < net24output["Prob"].size[0]; j++) {
				//cout << net24output["Prob"].data[j * 2 + 1] << endl;
				if (net24output["Prob"].data[j * 2 + 1] > min_confidence) {
					Rect2d this_rect = Rect2d(image_boxes[start_pos + j].x + image_boxes[start_pos + j].width * net24output["conv5-2"].data[j * 4 + 0],
						image_boxes[start_pos + j].y + image_boxes[start_pos + j].height * net24output["conv5-2"].data[j * 4 + 1],
						image_boxes[start_pos + j].width + image_boxes[start_pos + j].width * (net24output["conv5-2"].data[j * 4 + 2] - net24output["conv5-2"].data[j * 4 + 0]),
						image_boxes[start_pos + j].height + image_boxes[start_pos + j].height * (net24output["conv5-2"].data[j * 4 + 3] - net24output["conv5-2"].data[j * 4 + 1]));
					rect_and_scores.push_back(make_pair(this_rect, net24output["Prob"].data[j * 2 + 1]));
					//rects.push_back(this_rect);
					//scores.push_back(net24output["Prob"].data[j * 2 + 1]);
					//if (output_points) {
					//  vector<Point2d> point_list;
					//  for (int p = 0; p < 5; p++) {
					//    point_list.push_back(Point2d((net24output["conv5-2"].data[j * 10 + p * 2] + 12) / 24 * image_boxes[start_pos + j].width + image_boxes[start_pos + j].x,
					//      (net24output["conv5-2"].data[j * 10 + p * 2 + 1] + 12) / 24 * image_boxes[start_pos + j].height + image_boxes[start_pos + j].y));
					//  }
					//  allPoints.push_back(point_list);
					//}

				}
			}
		}
		//if (output_points) assert(allPoints.size() == rect_and_scores.size());
		vector<pair<Rect2d, float>> result;

		if (do_nms) {
			//std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
			vector<int> picked = nms_max(rect_and_scores, nms_threshold);
			for (auto& p : picked) {
				result.push_back(rect_and_scores[p]);
				//if (output_points) points.push_back(allPoints[p]);
			}
		}
		else {
			result = rect_and_scores;
		}
		return result;
	}

	vector<pair<Rect2d, float>> getNet48Final(vector<Mat> sub_images, vector<Rect2d> image_boxes, double min_confidence = 0.7,
		bool do_nms = true, double nms_threshold = 0.3,
		int batch_size = 500,
		bool output_points = false, vector<vector<Point2d>>& points = vector<vector<Point2d>>()) {
		int num = sub_images.size();
		if (num == 0) return vector<pair<Rect2d, float>>();
		assert(sub_images[0].rows == 48 && sub_images[0].cols == 48);
		vector<pair<Rect2d, float>> rect_and_scores;
		vector<vector<Point2d> > allPoints;

		int total_iter = ceil((double)num / (double)batch_size);
		for (int i = 0; i < total_iter; i++) {
			int start_pos = i * batch_size;
			if (i == total_iter - 1) batch_size = num - (total_iter - 1) * batch_size;
			vector<Mat> net_input = vector<Mat>(sub_images.begin() + start_pos, sub_images.begin() + start_pos + batch_size);
			auto net48output = kCaffeBinding->Forward(net_input, net48);
			for (int j = 0; j < net48output["Prob"].size[0]; j++) {
				//cout << net48output["Prob"].data[j * 2 + 1] << endl;
				if (net48output["Prob"].data[j * 2 + 1] > min_confidence) {
					Rect2d this_rect = Rect2d(image_boxes[start_pos + j].x + image_boxes[start_pos + j].width * net48output["conv6-2"].data[j * 4 + 0],
						image_boxes[start_pos + j].y + image_boxes[start_pos + j].height * net48output["conv6-2"].data[j * 4 + 1],
						image_boxes[start_pos + j].width + image_boxes[start_pos + j].width * (net48output["conv6-2"].data[j * 4 + 2] - net48output["conv6-2"].data[j * 4 + 0]),
						image_boxes[start_pos + j].height + image_boxes[start_pos + j].height * (net48output["conv6-2"].data[j * 4 + 3] - net48output["conv6-2"].data[j * 4 + 1]));
					rect_and_scores.push_back(make_pair(this_rect, net48output["Prob"].data[j * 2 + 1]));
					//rects.push_back(this_rect);
					//scores.push_back(net48output["conv6-3"].data[j * 2 + 1]);
					if (output_points) {
						vector<Point2d> point_list;
						for (int p = 0; p < 5; p++) {
							point_list.push_back(Point2d(net48output["conv6-3"].data[j * 10 + p] * image_boxes[start_pos + j].width + image_boxes[start_pos + j].x,
								net48output["conv6-3"].data[j * 10 + p + 5] * image_boxes[start_pos + j].height + image_boxes[start_pos + j].y));
						}
						allPoints.push_back(point_list);
					}

				}
			}
		}
		if (output_points) assert(allPoints.size() == rect_and_scores.size());
		vector<pair<Rect2d, float>> result;
		if (do_nms) {
			//std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
			vector<int> picked = nms_max(rect_and_scores, nms_threshold, IoU_MIN);
			for (auto& p : picked) {
				result.push_back(rect_and_scores[p]);
				if (output_points) points.push_back(allPoints[p]);
			}
			//cout << "nms time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - p1).count() / 1000 << "ms" << endl;
			//nms_avg(rects, scores, nms_threshold);
			//for (auto& p : rects) {
			//  result.push_back(make_pair<Rect2d, float>(Rect2d(p.x, p.y, p.width, p.height), 1.0f));
			//}
		}
		else {
			result = rect_and_scores;
		}
		return result;
	}

	vector<vector<Point2d>> GetFineLandmark(const Mat& input_image, vector<vector<Point2d>>& coarse_landmarks,
		vector<pair<Rect2d, float>>& face_rects, double width_factor = 0.25) {
		vector<Mat> sub_images;
		int face_num = face_rects.size();
		for (int n = 0; n < face_num; n++) {
			Mat concated_local_patch = Mat(24, 24, CV_8UC(15));
			vector<Mat> local_patches;
			double width = max(face_rects[n].first.width, face_rects[n].first.height) * width_factor;
			vector<int> from_to;
			from_to.reserve(30);
			for (int p = 0; p < 5; p++) {
				Mat local_patch_p = cropImage(input_image,
					Rect2d(coarse_landmarks[n][p].x - width / 2, coarse_landmarks[n][p].y - width / 2, width, width),
					Size(24, 24), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
				local_patch_p = local_patch_p.t();
				local_patches.push_back(local_patch_p);
				from_to.insert(from_to.end(), { p * 3 + 0, p * 3 + 2,
					p * 3 + 1, p * 3 + 1,
					p * 3 + 2, p * 3 + 0 });
			}
			mixChannels(local_patches, { concated_local_patch }, from_to);
			sub_images.push_back(concated_local_patch);
		}
		auto netLocOutput = kCaffeBinding->Forward(sub_images, netLoc);
		//for (int i = 0; i < netLocOutput.size(); i++) {
		//  cout << netLocOutput[i].name << endl;
		//}
		//cout << netLocOutput[0].name<< " size:" << netLocOutput[0].size[0] << " " << netLocOutput[0].size[1] << " " << netLocOutput[0].size[2] << " " << netLocOutput[0].size[3] << endl;
		for (int n = 0; n < face_num; n++) {
			double width = max(face_rects[n].first.width, face_rects[n].first.height) * width_factor;
			for (int p = 0; p < 5; p++) {
				coarse_landmarks[n][p].x = coarse_landmarks[n][p].x - width / 2 + netLocOutput["fc5_" + to_string(p + 1)].data[2 * n + 0] * width;
				coarse_landmarks[n][p].y = coarse_landmarks[n][p].y - width / 2 + netLocOutput["fc5_" + to_string(p + 1)].data[2 * n + 1] * width;
			}
		}
		return coarse_landmarks;
	}

	vector<pair<Rect2d, float>> GetDetection(const Mat& input_image, double start_scale = 1, double min_confidence = 0.995,
		bool do_nms = true, double nms_threshold = 0.7,
		bool output_points = false, vector<vector<Point2d>>& points = vector<vector<Point2d>>()) {
		if (points.size() > 0) points.clear();

		Mat clone_image = input_image.clone();//for drawing
		//std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
		auto proposal = getNet12ProposalAcc(clone_image, 0.6, start_scale, do_nms, nms_threshold);
		//std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
		//cout << "proposal time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;
		//cout << "proposal: " << proposal.size() << endl;
		if (proposal.size() == 0) return vector<pair<Rect2d, float>>();
		vector<Mat> sub_images;
		sub_images.reserve(proposal.size());
		vector<Rect2d> image_boxes;
		image_boxes.reserve(proposal.size());
		for (auto& p : proposal) {
			make_rect_square(p.first);
			//fixRect(p.first, input_image.size(), true);
			if (p.first.width < 9 || p.first.height < 9) continue;
			Mat sub_image = cropImage(input_image, p.first, Size(24, 24), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
			sub_images.push_back(sub_image);
			image_boxes.push_back(p.first);
		}
		std::chrono::time_point<std::chrono::system_clock> p2 = std::chrono::system_clock::now();
		//cout << "gen_list time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count() / 1000 << "ms" << endl;
		auto refined = getNet24Refined(sub_images, image_boxes, 0.7, do_nms, nms_threshold, 500);
		std::chrono::time_point<std::chrono::system_clock> p3 = std::chrono::system_clock::now();
		//cout << "refine time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p3 - p2).count() / 1000 << "ms" << endl;
		//cout << "refined: " << refined.size() << endl;
		if (refined.size() == 0) return vector<pair<Rect2d, float>>();
		//Mat image_show = input_image.clone();
		//for (auto& rect : refined) {
		//    rectangle(image_show, rect.first, Scalar(0, 255, 0), 1);
		//}

		//imshow("refined", image_show);
		//waitKey(0);

		vector<Mat> sub_images48;
		sub_images48.reserve(refined.size());
		vector<Rect2d> image_boxes48;
		image_boxes48.reserve(refined.size());
		//int start = 0;
		for (auto& p : refined) {
			make_rect_square(p.first);
			//fixRect(p.first, input_image.size(), true);
			if (p.first.width < 9 || p.first.height < 9) continue;
			Mat sub_image = cropImage(input_image, p.first, Size(48, 48), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
			sub_images48.push_back(sub_image);
			image_boxes48.push_back(p.first);
		}
		auto final = getNet48Final(sub_images48, image_boxes48, min_confidence, do_nms, nms_threshold, 500, output_points, points);
		//std::chrono::time_point<std::chrono::system_clock> p4 = std::chrono::system_clock::now();
		//cout << "final time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p4 - p3).count() / 1000 << "ms" << endl;
		//cout << "final: " << final.size() << endl;
		std::chrono::time_point<std::chrono::system_clock> p5 = std::chrono::system_clock::now();
		if (output_points && final.size() > 0) {
			GetFineLandmark(input_image, points, final);
			//p5 = std::chrono::system_clock::now();
			//cout << "fine landmark time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p5 - p4).count() / 1000 << "ms" << endl;
		}
		//cout<<"total time:"<< (float)std::chrono::duration_cast<std::chrono::microseconds>(p5 - p0).count() / 1000 << "ms" << endl;
		return final;
	}

	vector<pair<Rect2d, float>> ForceGetLandmark(Mat& input_image, Rect2d CoarseRect, vector<vector<Point2d>>& points = vector<vector<Point2d>>()) {
		make_rect_square(CoarseRect);
		Mat sub_image = cropImage(input_image, CoarseRect, Size(48, 48), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
		auto final = getNet48Final({ sub_image }, { CoarseRect }, 0, true, 0.7, 500, true, points);
		if (final.size() > 0) {
			GetFineLandmark(input_image, points, final);
		}
		return final;
	}

	int net12, net12_stitch, net24, net48, netLoc;
	float scale_decay_;
	int input_width, input_height;
};

MTCNN::MTCNN(const vector<char>& package, int gpu_id){
	this->_native = new CascadeCNN(package, gpu_id);
}

MTCNN::MTCNN(string net12_definition, string net12_weights,
	string net12_stitch_definition, string net12_stitch_weights,
	string net24_definition, string net24_weights,
	string net48_definition, string net48_weights,
	string netLoc_definition, string netLoc_weights,
	int gpu_id){

	this->_native = new CascadeCNN(
		net12_definition, net12_weights,
		net12_stitch_definition, net12_stitch_weights,
		net24_definition, net24_weights,
		net48_definition, net48_weights,
		netLoc_definition, netLoc_weights, gpu_id);
}

vector<pair<Rect, float>> MTCNN::detect2(const Mat& input_image, double start_scale, double min_confidence,
	bool do_nms, double nms_threshold, bool output_points, vector<vector<Point2d>>& points){

	CascadeCNN* p = (CascadeCNN*)this->_native;
	vector<pair<Rect2d, float>> r = p->GetDetection(input_image, start_scale, min_confidence, do_nms, nms_threshold, output_points, points);
	vector<pair<Rect, float>> out(r.size());
	for (int i = 0; i < r.size(); ++i)
		out[i] = make_pair(r[i].first, r[i].second);
	return out;
}

vector<Rect> MTCNN::detect(const Mat& input_image, int min_face, double min_confidence,
	bool do_nms, double nms_threshold,
	bool output_points, vector<vector<Point2d>>& points){

	double start_scale = 12.0 / min_face;
	CascadeCNN* p = (CascadeCNN*)this->_native;
	vector<pair<Rect2d, float>> r = p->GetDetection(input_image, start_scale, min_confidence, do_nms, nms_threshold, output_points, points);
	vector<Rect> out(r.size());
	for (int i = 0; i < r.size(); ++i)
		out[i] = r[i].first;
	return out;
}

MTCNN::FaceList* MTCNN::detect3(const Mat& input_image, double start_scale, double min_confidence,
	bool do_nms, double nms_threshold,
	bool output_points){
	CascadeCNN* p = (CascadeCNN*)this->_native;

	vector<vector<Point2d>> ps;
	vector<pair<Rect2d, float>> rcs = p->GetDetection(input_image, start_scale, min_confidence, do_nms, nms_threshold, output_points, ps);

	MTCNN::FaceList* flist = new MTCNN::FaceList();
	memset(flist, 0, sizeof(*flist));
	flist->count = rcs.size();
	flist->list = flist->count == 0 ? 0 : new FaceInfo[flist->count];

	for (int i = 0; i < rcs.size(); ++i){
		flist->list[i].bbox = rcs[i].first;
		flist->list[i].score = rcs[i].second;

		if (output_points){
			flist->list[i].pointCount = ps[i].size();
			flist->list[i].list = new Point2d[flist->list[i].pointCount];
			memcpy(flist->list[i].list, &ps[i][0], sizeof(Point2d) * ps[i].size());
		}
	}
	return flist;
}

void releaseFaceList(MTCNN::FaceList* flist){
	if (flist){
		if (flist->list){
			for (int i = 0; i < flist->count; ++i){
				if (flist->list[i].list)
					delete flist->list[i].list;
			}
			delete flist->list;
		}
		delete flist;
	}
}