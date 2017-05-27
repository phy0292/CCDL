
#include <cv.h>
#include <highgui.h>
#include <cv_import_static_lib.h>
#include <vector>
#include <string>
#include <fstream>

#include "classification.h"
#pragma comment(lib, "classification_dll.lib")

using namespace std;
using namespace cv;

vector<string> loadCodeMap(const char* file){
	ifstream infile(file);
	string line;
	vector<string> out;
	while (std::getline(infile, line)){
		out.push_back(line);
	}
	return out;
}

string getResult(const vector<string>& codeMap, SoftmaxResult* result){
	string out;
	for (int i = 0; i < result->count; ++i){
		out += codeMap[result->list[i].result[0].label];
	}
	return out;
}

void main(){

	vector<string> codeMap = loadCodeMap("码表.txt");
	Classifier c("deploy.prototxt", "nin_iter_16000.caffemodel");
	Mat im = imread("测试图片.png");
	WPtr<SoftmaxResult> result = c.predictSoftmax(im, 1);
	string code = getResult(codeMap, result);

	printf("识别结果: %s\n", code.c_str());
}