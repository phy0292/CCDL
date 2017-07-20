//#include <afx.h>
#include "liaojie_server.h"
#include "client_context.h"
#include "func.h"
#include <stdio.h>
#include <process.h>
#include <cv.h>
#include <highgui.h>
#include "classification.h"

//void* AFX_CDECL operator new(size_t nSize, LPCSTR lpszFileName, int nLine);
//#define new new(__FILE__, __LINE__)

using namespace std;
using namespace cv;
vector<Classifier*> classifiers;
vector<TaskPool*> pools;
char labelToChar(int label){
	if (label >= 0 && label <= 9) return label + '0';
	else return label - 10 + 'A';
}

void labelToStr(int* labels, int num, char* buf){
	for (int i = 0; i < num; ++i)
		buf[i] = labelToChar(labels[i]);
}

string vc2str(const vector<char>& value){
	vector<char> tmp = value;
	tmp.push_back(0);
	return &tmp[0];
}

int vc2int(const vector<char>& value){
	int v = 0;
	memcpy(&v, &value[0], 4);
	return v;
}

float vc2float(const vector<char>& value){
	float v = 0;
	memcpy(&v, &value[0], 4);
	return v;
}

vector<float> vc2floatList(const vector<char>& value){
	vector<float> arr;
	for (int i = 0; i < value.size(); i += 4){
		float v = 0;
		memcpy(&v, &value[i], 4);
		arr.push_back(v);
	}
	return arr;
}

struct param{
	string name;
	vector<char> value;

	param(const char* name, void* data, int len){
		this->name = name;
		this->value = vector<char>((char*)data, (char*)data + len);
	}

	param(const char* name, const char* value){
		this->name = name;
		int len = strlen(value);
		this->value = vector<char>((char*)value, (char*)value + len);
	}

	param(const char* name, int value){
		this->name = name;
		this->value = vector<char>((char*)&value, (char*)&value + sizeof(value));
	}

	param(const char* name, float value){
		this->name = name;
		this->value = vector<char>((char*)&value, (char*)&value + sizeof(value));
	}

	param(){

	}
};

struct cmd{
	string name;
	vector<param> params;
	map<string, param*> paramMap;
};

cmd* parseCmd(const void* data){
	cmd* c = new cmd();
	const char* p = (const char*)data;
	int n = 0;
	memcpy(&n, p, 4);  p += 4;

	c->name.resize(n);
	memcpy(&c->name[0], p, n);	p += n;

	memcpy(&n, p, 4); p += 4;
	if (n == 0)
		return c;

	c->params.resize(n);
	for (int i = 0; i < n; ++i){
		int slen = 0;
		memcpy(&slen, p, 4); p += 4;
		c->params[i].name.resize(slen);
		memcpy(&c->params[i].name[0], p, slen); p += slen;

		int value_len = 0;
		memcpy(&value_len, p, sizeof(value_len)); p += 4;
		if (value_len > 0){
			c->params[i].value.resize(value_len);
			memcpy(&c->params[i].value[0], p, value_len); p += value_len;
		}
	}

	for (int i = 0; i < c->params.size(); ++i)
		c->paramMap[c->params[i].name] = &c->params[i];
	return c;
}

void encodeToBuffer(const vector<param>& params, vector<char>& buf){
	int _lenbuf = 0;
	for (int i = 0; i < params.size(); ++i)
		_lenbuf += params[i].name.size() + 4 + params[i].value.size() + 4;

	buf.resize(_lenbuf + 4 + 4);
	char* p = &buf[0];
	int n = _lenbuf + 4;
	memcpy(p, &n, 4); p += 4;

	n = params.size();
	memcpy(p, &n, 4); p += 4;

	for (int i = 0; i < params.size(); ++i){
		n = params[i].name.size();
		memcpy(p, &n, 4); p += 4;
		memcpy(p, &params[i].name[0], params[i].name.size()); p += params[i].name.size();

		int num = params[i].value.size();
		memcpy(p, &num, 4); p += 4;

		if (num > 0){
			memcpy(p, &params[i].value[0], num); p += num;
		}
	}
}

void resp(clientContext* client, const vector<param>& params){
	vector<char> buf;
	encodeToBuffer(params, buf);
	//send(*client, &buf[0], buf.size(), 0);
	if (buf.size() > 0){
		client->send.buffer.buf = &buf[0];
		client->send.buffer.len = buf.size();
		client->postSendSignal();
		shutdown(client->socket, SD_SEND);
	}
}

string getParamStr(cmd* c, const char* name, const char* defaultValue = ""){
	if (c->paramMap.find(name) == c->paramMap.end()) return defaultValue;

	param* p = c->paramMap[name];
	if (p == 0) return defaultValue;
	return vc2str(p->value);
}

Mat getParamImage(cmd* c, const char* name, int flags){
	if (c->paramMap.find(name) == c->paramMap.end()) return Mat();

	param* p = c->paramMap[name];
	if (p == 0) return Mat();

	Mat img;
	try{
		img = cv::imdecode(p->value, flags);
	}
	catch (const cv::Exception& e){
		fcLog("error: %s", e.what());
	}
	catch (...){
		fcLog("image decode error.");
	}
	return img;
}

vector<float> getParamFloatList(cmd* c, const char* name){
	vector<float> arr;
	if (c->paramMap.find(name) == c->paramMap.end()) return arr;

	param* p = c->paramMap[name];
	if (p == 0) return arr;
	return vc2floatList(p->value);
}

int getParamInt(cmd* c, const char* name, int defaultValue = 0){
	if (c->paramMap.find(name) == c->paramMap.end()) return defaultValue;

	param* p = c->paramMap[name];
	if (p == 0) return defaultValue;
	return vc2int(p->value);
}

float getParamFloat(cmd* c, const char* name, float defaultValue = 0){
	if (c->paramMap.find(name) == c->paramMap.end()) return defaultValue;

	param* p = c->paramMap[name];
	if (p == 0) return defaultValue;
	return vc2float(p->value);
}

bool hasParam(cmd* c, const char* name){
	return c->paramMap.find(name) != c->paramMap.end();
}

void error(vector<param>& ps, const char* msg, int code = 0){
	ps.push_back(param("error", msg));
	ps.push_back(param("error_code", code));
}

void clearAllModel(){
	for (int i = 0; i < classifiers.size(); ++i){
		if (classifiers[i])
			delete classifiers[i];
	}
	classifiers.clear();
}

void stopServer(void* p){
	ljStop();
}

void execute(cmd* c, vector<param>& ps){
	if (c->name == "create_classifier"){
		vector<float> meanvalue = getParamFloatList(c, "meanvalue");
		if (meanvalue.size() == 0) meanvalue.push_back(0);

		Classifier* classf = new Classifier(
			getParamStr(c, "prototxt_file").c_str(),
			getParamStr(c, "caffe_model").c_str(),
			getParamFloat(c, "raw_scale", 1),
			getParamStr(c, "mean_file").c_str(),
			getParamInt(c, "meanvalue_dims", 0),
			&meanvalue[0],
			getParamInt(c, "gpu_id", -1),
			getParamInt(c, "cache_size", 1));

#if 0
		fcLog("meanvalue:");
		for (int i = 0; i < meanvalue.size(); ++i){
			fcLog("%f ", meanvalue[i]);
		}
		fcLog("\nmeandims: %d\nraw_scale: %f\n", getParamInt(c, "meanvalue_dims", 0), getParamFloat(c, "raw_scale", 1));
#endif

		classifiers.push_back(classf);
		ps.push_back(param("model_id", (int)classifiers.size()));
	}
	else if (c->name == "clear_models"){
		int modelID = getParamInt(c, "model_id");
		if (modelID < 1 || modelID > classifiers.size()){
			error(ps, "错误的模型id");
			return;
		}

		modelID--;
		if (classifiers[modelID] != 0)
			delete classifiers[modelID];

		classifiers[modelID] = 0;
	}
	else if (c->name == "extractFeature"){
		int modelID = getParamInt(c, "model_id");
		if (modelID < 1 || modelID > classifiers.size()){
			error(ps, "错误的模型id");
			return;
		}

		if (!hasParam(c, "feature_layer")){
			error(ps, "必须指定feature_layer，作为提取层的名字");
			return;
		}

		string featureLayer = getParamStr(c, "feature_layer");

		modelID--;
		if (classifiers[modelID] == 0){
			error(ps, "该模型已经被干掉了");
			return;
		}

		Mat im = getParamImage(c, "image", getParamInt(c, "isColor", 1));
		if (im.empty()){
			error(ps, "图像解析失败");
			return;
		}

		bool isScale = getParamInt(c, "isScale");
		if (isScale){
			im.convertTo(im, CV_32F);
			im = im / 255;
		}

		Classifier* cf = classifiers[modelID];
		WPtr<BlobData> result = cf->extfeature(im, featureLayer.c_str());
		if (!result){
			error(ps, "发生错误");
			return;
		}
		ps.push_back(param("feature", result->list, result->count * sizeof(result->list[0])));
	}
	else if (c->name == "predictSoftmax"){
		int modelID = getParamInt(c, "model_id");
		int top_n = getParamInt(c, "top_n", 1);
		if (modelID < 1 || modelID > classifiers.size()){
			error(ps, "错误的模型id");
			return;
		}

		modelID--;
		if (classifiers[modelID] == 0){
			error(ps, "该模型已经被干掉了");
			return;
		}

		Mat im = getParamImage(c, "image", getParamInt(c, "isColor", 1));
		if (im.empty()){
			error(ps, "图像解析失败");
			return;
		}

		bool isScale = getParamInt(c, "isScale");
		if (isScale){
			im.convertTo(im, CV_32F);
			im = im / 255;
		}

		Classifier* cf = classifiers[modelID];
		SoftmaxResult* result = cf->predictSoftmax(im, top_n);
		if (!result){
			error(ps, "发生错误");
			return;
		}

		vector<int> labs;
		vector<float> confs;
		for (int i = 0; i < result->count; ++i){
			for (int j = 0; j < result->list[i].count; ++j){
				labs.push_back(result->list[i].result[j].label);
				confs.push_back(result->list[i].result[j].conf);
			}
		}

		ps.push_back(param("labs", &labs[0], labs.size()*sizeof(labs[0])));
		ps.push_back(param("confs", &confs[0], confs.size()*sizeof(confs[0])));
		releaseSoftmaxResult(result);
	}
	else if (c->name == "shutdown"){
		clearAllModel();
		_beginthread(&stopServer, 0, 0);
		//exit(0);
	}
	else if (c->name == "createTaskPool"){
		vector<float> meanvalue = getParamFloatList(c, "meanvalue");
		if (meanvalue.size() == 0) meanvalue.push_back(0);

		TaskPool* pool = createTaskPool(
			getParamStr(c, "prototxt_file").c_str(),
			getParamStr(c, "caffe_model").c_str(),
			getParamFloat(c, "raw_scale", 1),
			getParamStr(c, "mean_file").c_str(),
			getParamInt(c, "meanvalue_dims", 0),
			&meanvalue[0],
			getParamInt(c, "gpu_id", -1),
			getParamInt(c, "batch_size", 1));

		pools.push_back(pool);
		ps.push_back(param("model_id", (int)pools.size()));
	}
	else if (c->name == "predictSoftmaxByPool"){
		int modelID = getParamInt(c, "model_id");
		int top_n = getParamInt(c, "top_n", 1);
		if (modelID < 1 || modelID > pools.size()){
			error(ps, "错误的模型id");
			return;
		}

		modelID--;
		if (pools[modelID] == 0){
			error(ps, "该模型已经被干掉了");
			return;
		}

		Mat im = getParamImage(c, "image", getParamInt(c, "isColor", 1));
		if (im.empty()){
			error(ps, "图像解析失败");
			return;
		}

		bool isScale = getParamInt(c, "isScale");
		if (isScale){
			im.convertTo(im, CV_32F);
			im = im / 255;
		}

		TaskPool* pool = pools[modelID];
		SoftmaxResult* result = predictSoftmaxByTaskPool2(pool, &im, top_n);
		if (!result){
			error(ps, "发生错误");
			return;
		}

		vector<int> labs;
		vector<float> confs;
		for (int i = 0; i < result->count; ++i){
			for (int j = 0; j < result->list[i].count; ++j){
				labs.push_back(result->list[i].result[j].label);
				confs.push_back(result->list[i].result[j].conf);
			}
		}

		ps.push_back(param("labs", &labs[0], labs.size()*sizeof(labs[0])));
		ps.push_back(param("confs", &confs[0], confs.size()*sizeof(confs[0])));
		releaseSoftmaxResult(result);
	}
	else if (c->name == "forwardByTaskPool"){
		int modelID = getParamInt(c, "model_id");
		string blob_name = getParamStr(c, "blob_name");
		if (modelID < 1 || modelID > pools.size()){
			error(ps, "错误的模型id");
			return;
		}

		modelID--;
		if (pools[modelID] == 0){
			error(ps, "该模型已经被干掉了");
			return;
		}

		Mat im = getParamImage(c, "image", getParamInt(c, "isColor", 1));
		if (im.empty()){
			error(ps, "图像解析失败");
			return;
		}

		bool isScale = getParamInt(c, "isScale");
		if (isScale){
			im.convertTo(im, CV_32F);
			im = im / 255;
		}

		TaskPool* pool = pools[modelID];
		BlobData* result = forwardByTaskPool2(pool, &im, blob_name.c_str());
		if (!result){
			error(ps, "发生错误");
			return;
		}

		vector<float> confs(result->list, result->list + result->count);
		ps.push_back(param("width", result->width));
		ps.push_back(param("height", result->height));
		ps.push_back(param("channels", result->channels));
		ps.push_back(param("count", result->count));
		ps.push_back(param("blob_data", &confs[0], confs.size()*sizeof(confs[0])));
		releaseBlobData(result);
	}
}

void doAction(clientContext* client){
	cmd* c = parseCmd(&client->package->at(0));
	//fcLog("execmd: %s[%d]", c->name.c_str(), c->params.size());

	vector<param> ps;
	execute(c, ps);
	resp(client, ps);
	delete c;
}

void doRecvData(clientContext* client, int dwBytes){
	char* data = client->recv.data;
	int len = dwBytes;
	while (len > 0){
		if (client->packageType == packageBegin){
			if (len < 4) return;

			memcpy(&client->packageRemainLength, data, 4);
			if (client->packageRemainLength <= 0)
				return;

			len -= 4;
			data += 4;
			client->packageType = packageBuild;
			client->package->clear();
			client->package->reserve(client->packageRemainLength);
		}

		int getLen = min(len, client->packageRemainLength);
		if (getLen > 0)
			client->package->insert(client->package->end(), data, data + getLen);
		
		len -= getLen;
		client->packageRemainLength -= getLen;
		data += getLen;

		if (client->packageRemainLength == 0){
			doAction(client);
			client->packageType = packageBegin;
		}
	}
}

int main(int argc, char** argv)
{
#if 0
	int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag(tmpFlag);
#endif

	int defaultPort = 16988;
	int defaultnum_worker_threads = -1;
	printf("usage: server.exe port[%d] num_worker_threads[%d]\n\n", defaultPort, defaultnum_worker_threads);

	int port = argc > 1 ? atoi(argv[1]) : defaultPort;
	int num_threads = argc > 2 ? atoi(argv[2]) : defaultnum_worker_threads;

	if(!ljInitialize()){
		fcLog("Server initialize error: %s", ljErrorMessage());
		return 0;
	}
	
	setDoRecvDataListener(doRecvData);
	if (!ljSetup(port, -1, num_threads)){
		fcLog("Setup error.");
		goto error_exit;
	}

	fcLog("loop.");
	ljLoop();

	fcLog("server shutdown.");
error_exit:
	ljStop();
	ljUninitialize();
	return 0;
}