#include "classification.h"
#include <highgui.h>
#include <process.h>
#define RootDir "../../../../../demo-data/veriCode/"

using namespace cv;

vector<char> readAtFile(const char* filename){
	FILE* f = fopen(filename, "rb");
	if (f == 0) return vector<char>();

	fseek(f, 0, SEEK_END);
	int len = ftell(f);
	fseek(f, 0, SEEK_SET);

	vector<char> buf(len);
	fread(&buf[0], 1, len, f);
	fclose(f);
	return buf;
}

void recThread(void* param){
	TaskPool* pool = (TaskPool*)param;
	vector<char> imd = readAtFile(RootDir "samples/00W0_27c86a8b9ce8d0b1fe1d3d47b4040a28.png");

	for(int i = 0; i < 1000; ++i){
		int labels[4];
		float confs[4];
		SoftmaxResult* val = predictSoftmaxByTaskPool(pool, &imd[0], imd.size(), 1);
		getMultiLabel(val, labels);
		getMultiConf(val, confs);
		if (i % 50 == 0){
			printf("labels = %d, %d, %d, %d\n", labels[0], labels[1], labels[2], labels[3]);
			printf("confs = %f, %f, %f, %f\n", confs[0], confs[1], confs[2], confs[3]);
		}
		releaseSoftmaxResult(val);
	}
}

#if 0
void main(int argc, char** argv){

#if 0
	Classifier c(RootDir "deploy.prototxt", RootDir "nin_iter_16000.caffemodel", 1.0, "", 0, 0, -1, 10);

	int num = argc > 1 ? atoi(argv[1]) : 1;
	Mat im = imread(RootDir "samples/00W0_27c86a8b9ce8d0b1fe1d3d47b4040a28.png");
	Mat im1 = imread(RootDir "samples/0FAW_a103991142caf37bfc7912c7cd2162b9.png");

	//单图测试
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
#if 0
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
		SoftmaxResult* val = softmax->list[i];
		getMultiLabel(val, labels);
		getMultiConf(val, confs);
		printf("labels = %d, %d, %d, %d\n", labels[0], labels[1], labels[2], labels[3]);
		printf("confs = %f, %f, %f, %f\n", confs[0], confs[1], confs[2], confs[3]);
	}
#endif
#endif

	//测试任务池
	TaskPool* pool = createTaskPool(RootDir "deploy.prototxt", RootDir "nin_iter_16000.caffemodel", 1.0, "", 0, 0, 0, 100);
	for (int i = 0; i < 8; ++i){
		_beginthread(recThread, 0, pool);
	}
	Sleep(100 * 1000);
	printf("停止...\n");
	releaseTaskPool(pool);
	printf("已经停止...\n");
	Sleep(3000);
}
#endif

typedef int(__stdcall *procCCTrainEventCallback)(int event, int param1, float param2, void* param3);
extern void setTrainEventCallback(procCCTrainEventCallback callback);
extern "C" Caffe_API void __stdcall setTraindEventCallback(procCCTrainEventCallback callback);
extern "C" Caffe_API int __stdcall train_network(char* args);

int __stdcall testx(int event, int param1, float param2, void* param3){
	printf("event %d:\n", event);
	if (event == 7){
		char* p = (char*)param3;
		p += 16;
		char** ptr = *(char***)p;
		printf("%s\n", ptr[0]);
	}
	return 0;
}
void main(){
	setTraindEventCallback(testx);

	string info = "train --solver=solver.prototxt";
	train_network((char*)info.c_str());
}