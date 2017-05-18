#include "classification.h"
#include <highgui.h>

#define RootDir "../../../../../demo-data/veriCode/"

using namespace cv;
void main(int argc, char** argv){

	Classifier c(RootDir "deploy.prototxt", RootDir "nin_iter_16000.caffemodel", 1.0, "", 0, 0, -1, 10);

	int num = argc > 1 ? atoi(argv[1]) : 1;
	Mat im = imread(RootDir "samples/00W0_27c86a8b9ce8d0b1fe1d3d47b4040a28.png");
	Mat im1 = imread(RootDir "samples/0FAW_a103991142caf37bfc7912c7cd2162b9.png");

#if 0
	WPtr<SoftmaxResult> softmax = c.predictSoftmax(im, 1);
	int labels[4];
	float confs[4];
	getMultiLabel(softmax, labels);
	getMultiConf(softmax, confs);
	printf("labels = %d, %d, %d, %d\n", labels[0], labels[1], labels[2], labels[3]);
	printf("confs = %f, %f, %f, %f\n", confs[0], confs[1], confs[2], confs[3]);
#endif

	///测试多图传入问题
#if 1
	vector<Mat> imgs;
	for (int i = 0; i <num; ++i){
		imgs.push_back(i % 2 == 0 ? im : im1);
	}

	double tck = cv::getTickCount();
	WPtr<MultiSoftmaxResult> softmax = c.predictSoftmax(imgs, 1);
	tck = (cv::getTickCount() - tck) / cv::getTickFrequency() * 1000.0;
	printf("耗时：%.2f ms\n", tck);
	for (int i = 0; i < softmax->count; ++i){
		int labels[4];
		float confs[4];
		SoftmaxResult* val = &softmax->list[i];
		getMultiLabel(val, labels);
		getMultiConf(val, confs);
		printf("labels = %d, %d, %d, %d\n", labels[0], labels[1], labels[2], labels[3]);
		printf("confs = %f, %f, %f, %f\n", confs[0], confs[1], confs[2], confs[3]);
	}
#endif
}