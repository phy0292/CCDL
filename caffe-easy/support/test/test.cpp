#ifdef BuildExe

#if 0
#include <classification-c.h>
#if 0
void main0(){

	const char* prototxt = "E:/project/3.github/CCDL/caffe-lite/wish/test/测试数据/deploy.prototxt";
	const char* model = "E:/project/3.github/CCDL/caffe-lite/wish/test/测试数据/nin_iter_90000.caffemodel";

	size_t _len_proto, _len_model;
	char* data_prototxt = (char*)paReadFile(prototxt, &_len_proto);
	char* data_model = (char*)paReadFile(model, &_len_model);
	Classifier* cls = createClassifierByData(data_prototxt, _len_proto, data_model, _len_model);
	printf("cls = %p\n", cls);

}

#endif

extern "C" bool __stdcall model_compress(const char* infile, const char* fileOutPath, int upLevel, int saveToNormal);

void main(){
	const char* proto = "E:/project/3.github/CCDL/caffe-lite/wish/test/模型压缩/deploy.prototxt";
	const char* model = "E:/project/3.github/CCDL/caffe-lite/wish/test/模型压缩/nin_iter_16000.caffemodel";
	const char* modelOut = "E:/project/3.github/CCDL/caffe-lite/wish/test/模型压缩/jg.ys.model";
	createClassifierByData("aa", 2, "22", 2);
	//model_compress(model, modelOut, 3, 0);
}

#endif


#include <mtcnn.h>
#include <map>
#include <string>
#include <vector>
#include <cv.h>
#include <highgui.h>
using namespace cv;
using namespace std;

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

void main(int argc, char** argv){

#if 0
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
#endif

	int gpu_id = -1;
	int min_face = 80;
	const char* package = "../../../../../demo-data/mtcnn.package";
	if (argc > 1) package = argv[1];
	if (argc > 2) min_face = atoi(argv[2]);
	if (argc > 3) gpu_id = atoi(argv[3]);

	VideoCapture cap(0); 
	Mat frame;
	MTCNN mtcnn(readAtFile(package), gpu_id);
	double fps = 0;
	double countTimes = 0;
	int fs = 0;

	printf("gpu_id = %d\n", gpu_id);

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
		printf("耗时：%.2f ms\n", tmp);
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
}
#endif