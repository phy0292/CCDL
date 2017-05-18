#include <map>
#include <string>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include "mtcnn.h"
#include "import-staticlib.h"
using namespace cv;
using namespace std;


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

int main(int argc, char** argv){
	int gpu_id = -1;		//-1为使用CPU模式，大于-1为GPU的编号
	int min_face = 80;
	const char* package = "../../../../../demo-data/mtcnn.package";
	if (argc > 1) package = argv[1];
	if (argc > 2) min_face = atoi(argv[2]);
	if (argc > 3) gpu_id = atoi(argv[3]);

	vector<char> data = readAtFile(package);
	if (data.size() == 0){
		printf("使用的GPUid是%d，最小人脸%d，模型文件：%s\n", gpu_id, min_face, package);
		printf("模型文件不存在：%s\n请到这里下载：http://www.zifuture.com/fs/4.mtcnn/mtcnn.package\n", package);
		return 0;
	}

	MTCNN mtcnn(data, gpu_id);
	VideoCapture cap(0); 
	Mat frame;
	double fps = 0;
	double countTimes = 0;
	int fs = 0;

	printf("使用的GPUid是%d，最小人脸%d，模型文件：%s\n", gpu_id, min_face, package);

	cap >> frame;
	while (!frame.empty()){
		double tmp = cv::getTickCount();
		vector<vector<Point2d>> keys;
		vector<Rect> boxs = mtcnn.detect(frame, min_face, 0.7, true, 0.7, true, keys);
		for (int i = 0; i < boxs.size(); ++i){
			rectangle(frame, boxs[i], Scalar(0, 255), 2);

			for (int k = 0; k < keys[i].size(); ++k){
				circle(frame, keys[i][k], 3, Scalar(0, 0, 255), -1);
			}
		}

		tmp = (cv::getTickCount() - tmp) / cv::getTickFrequency() * 1000.0;
		//printf("耗时：%.2f ms\n", tmp);
		countTimes += tmp;
		fs++;

		if (countTimes >= 1000){
			fps = countTimes / fs;

			countTimes = 0;
			fs = 0;
		}
		putText(frame, format("%.2f fps", fps), Point(10, 30), 1, 1, Scalar(0, 255), 1);
		imshow("esc to exit.", frame);

		//ESC
		if (waitKey(1) == 0x1B) break;
		cap >> frame;
	}
	return 0;
}



////////////////////////////////////////////////////////////////////
/////下面代码实现模型的打包操作
#if 0
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

bool encodePackage(const map<string, vector<char>>& package, const char* filename){
	FILE* f = fopen(filename, "wb");
	if (f == 0) return false;

	auto itr = package.begin();
	for (; itr != package.end(); ++itr){
		int len = itr->first.size();
		fwrite(&len, 1, sizeof(len), f);
		fwrite(&itr->first[0], 1, len, f);

		len = itr->second.size();
		fwrite(&len, 1, sizeof(len), f);
		fwrite(&itr->second[0], 1, len, f);
	}
	fclose(f);
	return true;
}

bool readAddToPackage(map<string, vector<char>>& package, const char* filename, const char* name){
	FILE* f = fopen(filename, "rb");
	if (f == 0) return false;

	fseek(f, 0, SEEK_END);
	int len = ftell(f);
	fseek(f, 0, SEEK_SET);

	vector<char> buf(len);
	fread(&buf[0], 1, len, f);
	fclose(f);
	package[name] = buf;
	return true;
}

int main(int argc, char** argv){
	//下面的代码把模型给打包
	map<string, vector<char>> package;
#define model_folder "E:/project/3.github/CCDL/feature/mtcnn/model/"
#define pack(filename, name)		readAddToPackage(package, model_folder ## filename, name)
	pack("det1-memory.prototxt", "det1.proto");
	pack("det1.caffemodel", "det1.model");
	pack("det1-memory-stitch.prototxt", "det1-stitch.proto");

	pack("det2-memory.prototxt", "det2.proto");
	pack("det2.caffemodel", "det2.model");

	pack("det3-memory.prototxt", "det3.proto");
	pack("det3.caffemodel", "det3.model");

	pack("det4-memory.prototxt", "det4.proto");
	pack("det4.caffemodel", "det4.model");

	encodePackage(package, "mtcnn.package");
	map<string, vector<char>> mp = prasePackage(readAtFile("mtcnn.package"));
}
#endif