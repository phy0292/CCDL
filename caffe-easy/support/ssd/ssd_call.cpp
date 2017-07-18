

#include <cv.h>
#include <highgui.h>
#include <classification.h>
#include  <iostream>
#include  <fstream>
#include  < string >
#include "pa_draw.h"

using namespace cv;
using namespace std;

struct DetectObjectInfo{
	int image_id;
	int label;
	float score;
	int xmin;
	int ymin;
	int xmax;
	int ymax;

	Rect rect(){
		return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
	}
};

vector<string> loadLabels(const char* labelsFile){
	ifstream fin(labelsFile);
	vector<string> out;

	string  s;
	while (fin >> s)
		out.push_back(s);

	return out;
}

vector<DetectObjectInfo> toDetInfo(BlobData* fr, int imWidth = 1, int imHeight = 1){
	vector<DetectObjectInfo> out;
	float* data = fr->list;
	for (int i = 0; i < fr->count; i += 7, data += 7){
		DetectObjectInfo obj;

		//if invalid det
		if (data[0] == -1)
			continue;

		obj.image_id = data[0];
		obj.label = data[1];
		obj.score = data[2];
		obj.xmin = data[3] * imWidth;
		obj.ymin = data[4] * imHeight;
		obj.xmax = data[5] * imWidth;
		obj.ymax = data[6] * imHeight;
		out.push_back(obj);
	}
	return out;
}

Scalar getColor(int label){
	static vector<Scalar> colors;
	if (colors.size() == 0){
#if 0
		for (float r = 127.5; r <= 256 + 127.5; r += 127.5){
			for (float g = 256; g >= 0; g -= 127.5){
				for (float b = 0; b <= 256; b += 127.5)
					colors.push_back(Scalar(b, g, r > 256 ? r - 256 : r));
			}
		}
#endif
		colors.push_back(Scalar(255, 0, 0));
		colors.push_back(Scalar(0, 255, 0));
		colors.push_back(Scalar(0, 0, 255));
		colors.push_back(Scalar(0, 255, 255));
		colors.push_back(Scalar(255, 0, 255));
		colors.push_back(Scalar(128, 0, 255));
		colors.push_back(Scalar(128, 255, 255));
		colors.push_back(Scalar(255, 128, 255));
		colors.push_back(Scalar(128, 255, 128));
	}
	return colors[label % colors.size()];
}

bool fileExists(const char* file){
	FILE* f = fopen(file, "rb");
	if (f == 0) return false;
	fclose(f);
	return true;
}

#if 1
int main(int argc, char** argv){
	//disableErrorOutput();
	char caffemodel[260] = "../../../../../demo-data/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel";
	char prototxt[260] = "../../../../../demo-data/SSD_300x300/deploy.prototxt";
	char labels[260] = "../../../../../demo-data/SSD_300x300/labels.txt";
	char test_image[260] = "../../../../../demo-data/test.jpg";
	if (argc > 1){
		const char* demoDataDir = argv[1];
		sprintf(caffemodel, "%s/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel", demoDataDir);
		sprintf(prototxt, "%s/SSD_300x300/deploy.prototxt", demoDataDir);
		sprintf(labels, "%s/SSD_300x300/labels.txt", demoDataDir);
		sprintf(test_image, "%s/test.jpg", demoDataDir);
	}

	if (!fileExists(caffemodel)){
		printf(
			"使用方法：ssd_call demo-data\n\n"
			"无法载入模型文件：%s\n"
			"请到这里下载模型：http://www.zifuture.com/fs/2.ssd/VGG_coco_SSD_300x300_iter_400000.caffemodel\n"
			"或者压缩过的模型：http://www.zifuture.com/fs/2.ssd/VGG_coco_SSD_300x300_iter_400000.ys.rar\n"
			"其他相关，请参考ssd章节的介绍：https://github.com/weiliu89/caffe/tree/ssd\n"
			, caffemodel);
		return 0;
	}

	vector<string> labelMap = loadLabels(labels);
	Mat im = imread(test_image);

	float means[] = { 104.0f, 117.0f, 123.0f };
	Classifier cc(prototxt, caffemodel, 1, 0, 3, means);
	WPtr<BlobData> fr = cc.extfeatureImgs({ im, im }, "detection_out");
	vector<DetectObjectInfo> objs = toDetInfo(fr, im.cols, im.rows);
	printf("%d, %d, %d, %d, count = %d\n", fr->count, fr->channels, fr->height, fr->width, fr->count);
	for (int i = 0; i < objs.size(); ++i){
		auto obj = objs[i];
		if (obj.score > 0.25){
			rectangle(im, obj.rect(), getColor(obj.label), 2);
			paDrawString(im, labelMap[obj.label].c_str(), Point(obj.xmin, obj.ymin - 20), getColor(obj.label), 20, true);
			printf("%s: %f\n", labelMap[obj.label].c_str(), obj.score);
		}
	}
	imshow("demo", im);
	waitKey();
	return 0;
}
#endif

#if 0
int main(int argc, char** argv){
	//disableErrorOutput();
	char caffemodel[260] = "../../../../../demo-data/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel";
	char prototxt[260] = "../../../../../demo-data/SSD_300x300/deploy.prototxt";
	char labels[260] = "../../../../../demo-data/SSD_300x300/labels.txt";
	char test_image[260] = "../../../../../demo-data/test.jpg";
	if (argc > 1){
		const char* demoDataDir = argv[1];
		sprintf(caffemodel, "%s/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel", demoDataDir);
		sprintf(prototxt, "%s/SSD_300x300/deploy.prototxt", demoDataDir);
		sprintf(labels, "%s/SSD_300x300/labels.txt", demoDataDir);
		sprintf(test_image, "%s/test.jpg", demoDataDir);
	}

	if (!fileExists(caffemodel)){
		printf(
			"使用方法：ssd_call demo-data\n\n"
			"无法载入模型文件：%s\n"
			"请到这里下载模型：http://www.zifuture.com/fs/2.ssd/VGG_coco_SSD_300x300_iter_400000.caffemodel\n"
			"或者压缩过的模型：http://www.zifuture.com/fs/2.ssd/VGG_coco_SSD_300x300_iter_400000.ys.rar\n"
			"其他相关，请参考ssd章节的介绍：https://github.com/weiliu89/caffe/tree/ssd\n"
			, caffemodel);
		return 0;
	}

	vector<string> labelMap = loadLabels(labels);
	float means[] = { 104.0f, 117.0f, 123.0f };
	Classifier cc(prototxt, caffemodel, 1, 0, 3, means, 0);
	VideoCapture cap(0);
	Mat frame;

	cap >> frame;
	while (!frame.empty()){
		WPtr<BlobData> fr = cc.extfeature(frame, "detection_out");
		vector<DetectObjectInfo> objs = toDetInfo(fr, frame.cols, frame.rows);

		for (int i = 0; i < objs.size(); ++i){
			auto obj = objs[i];
			if (obj.score > 0.25){
				rectangle(frame, obj.rect(), getColor(obj.label), 2);
				paDrawString(frame, labelMap[obj.label].c_str(), Point(obj.xmin, obj.ymin - 20), getColor(obj.label), 20, true);
				//printf("%s: %f\n", labelMap[obj.label].c_str(), obj.score);
			}
		}
		imshow("demo", frame);
		waitKey(1);
		cap >> frame;
	}
	return 0;
}
#endif