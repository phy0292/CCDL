
//这是一个lstm+cnn的ocr例子
//2017年7月14日 12:23:54
//wish

#include "caffe.pb.h"
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include "classification-c.h"
#include <thread>
#include <io.h>
#include <fcntl.h>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
using namespace std;

#ifdef _DEBUG
#pragma comment(lib, "libprotobufd.lib")
#else
#pragma comment(lib, "libprotobuf.lib")
#endif
#pragma comment(lib, "classification_dll.lib")

vector<char> readFile(const char* file){
	vector<char> data;
	FILE* f = fopen(file, "rb");
	if (!f) return data;

	int len = 0;
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);

	if (len > 0){
		data.resize(len);
		fread(&data[0], 1, len, f);
	}
	fclose(f);
	return data;
}

void loadCodeMap(const char* file, vector<string> & out){
	ifstream infile(file);
	string line;
	while (std::getline(infile, line)){
		out.push_back(line);
	}
}

string getLabel(const vector<string>& labelMap, int index){
	if (index < 0 || index >= labelMap.size())
		return "*";

	return labelMap[index];
}

int argmax(float* arr, int begin, int end)
{
	try
	{
		int mxInd = 0;
		float acc = -9999;
		for (int i = begin; i < end; i++)
		{
			if (acc < arr[i])
			{
				mxInd = i;
				acc = arr[i];
			}
		}
		return mxInd - begin;
	}
	catch (exception)
	{

		return -1;
	}

}
CRITICAL_SECTION  g_csThreadCode;
HANDLE            g_hThreadParameter;
long lReleaseCount = 0;
const int MAX_THREAD = 90;
int 正确 = 0, 错误 = 0;
int num_output = 0;
int time_step = 0;
vector<string> labelMap;
TaskPool* classifierHandle;
string strPath;
void doproc(const string strFile)
{
	string strFileName = strFile;
	string strFileNameAll = strPath + strFileName;
	//printf("开始一个线程%s", strFileName.c_str());
	vector<char> data = readFile(strFileNameAll.c_str());
	BlobData* 张量句柄;
	for (int i = 0; i < 30; i++)
	{
		张量句柄 = forwardByTaskPool(classifierHandle, &data[0], data.size(), "premuted_fc");
		if (张量句柄 != NULL)break;
		Sleep(30);
	}
	if (张量句柄 != NULL)
	{
		int 空白标签索引 = num_output - 1;
		int prev = 空白标签索引;
		int o = 0;
		string rt = "";
		int len = getBlobLength(张量句柄);
		if (len != 0)
		{
			float* permute_fc = new float[len];
			try
			{
				cpyBlobData(permute_fc, 张量句柄);

				for (int i = 1; i < time_step; i++)
				{
					o = argmax(permute_fc, (i - 1) * num_output, i * num_output);

					if (o != 空白标签索引 && prev != o && o > -1 && o < num_output)
					{
						rt += labelMap[o];
					}

					prev = o;
				}
			}
			catch (...)
			{
			}
			delete[] permute_fc;
			string strS = strFileName.substr(0, strFileName.find('_', 0));
			string s;
			EnterCriticalSection(&g_csThreadCode);
			if (strS == rt)
			{
				正确++;
				s = "正确";
			}
			else
			{
				错误++;
				s = "错误";
				printf("识别的结果是：%s\t%s\t正确答案=%s\n", rt.c_str(), s.c_str(), strFileName.c_str());
			}
			LeaveCriticalSection(&g_csThreadCode);
		}
		releaseBlobData(张量句柄);
	}
	ReleaseSemaphore(g_hThreadParameter, 1, &lReleaseCount);
}

void FindFile()
{
	WIN32_FIND_DATAA  findData = { 0 };
	string strFindPath = strPath + "*.bmp";
	//查找第一个文件  
	HANDLE hFindFine = FindFirstFileA(strFindPath.c_str(), &findData);
	if (INVALID_HANDLE_VALUE == hFindFine)
		return;
	InitializeCriticalSection(&g_csThreadCode);
	g_hThreadParameter = CreateSemaphore(NULL, MAX_THREAD, MAX_THREAD, NULL);
	
	//循环查找文件
	int i = 0;
	do
	{
		WaitForSingleObject(g_hThreadParameter, INFINITE);
		//string str = findData.cFileName;
		thread(doproc, findData.cFileName).detach();
		//Sleep(1);
		//doproc(findData.cFileName);
	} while (FindNextFileA(hFindFine, &findData));
	while (lReleaseCount < MAX_THREAD - 1)
	{
		Sleep(100);
	}
	DeleteCriticalSection(&g_csThreadCode);
	CloseHandle(g_hThreadParameter);
	printf("正确=%d个，错误=%d个，正确率=%f\n", 正确, 错误, (double)正确 / (错误 + 正确));
	//关闭文件搜索句柄  
	FindClose(hFindFine);
}

//获取deploy.prototxt网络中的num_output和time_step参数值
bool __stdcall GetProtoParam2(const char* filename){
	caffe::NetParameter proto;
	int fd = open(filename, O_RDONLY);
	if (fd == -1)return false;
	google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
	bool success = google::protobuf::TextFormat::Parse(input, &proto);
	delete input;
	close(fd);
	num_output = 0;
	time_step = 0;
	if (success)
	{
		for (int i = 0; i < proto.layer_size(); i++)
		{
			caffe::LayerParameter layerp = proto.layer(i);
			if (layerp.name() == "fc1x")
			{
				num_output = layerp.inner_product_param().num_output();
			}
			if (layerp.name() == "indicator")
			{
				time_step = layerp.continuation_indicator_param().time_step();
			}
			if (num_output != 0 && time_step != 0)break;
		}
	}
	return num_output != 0 && time_step != 0;
}
void FindFile2(const std::string& strPath, Classifier* classifierHandle)
{

	WIN32_FIND_DATAA  findData = { 0 };
	string strFindPath = strPath + "*.bmp";
	//查找第一个文件  
	HANDLE hFindFine = FindFirstFileA(strFindPath.c_str(), &findData);
	if (INVALID_HANDLE_VALUE == hFindFine)
		return;
	//循环递归查找文件  
	int 正确 = 0, 错误 = 0;
	do
	{
		vector<char> data = readFile((strPath + findData.cFileName).c_str());
		BlobData* 张量句柄;
		for (int i = 0; i < 30; i++)
		{
			forward(classifierHandle, &data[0], data.size());// , "premuted_fc");
			张量句柄 = getBlobData(classifierHandle, "premuted_fc");
			if (张量句柄 != NULL)break;
			Sleep(30);
		}
		if (张量句柄 != NULL)
		{
			//int time_step = 19;// 15
			int 空白标签索引 = num_output - 1; //25;//34; //这个表示最大的字符数，包括下划线的空白符
			//int 字符总数 = 空白标签索引 + 1;
			int prev = 空白标签索引;
			int o = 0;
			string rt = "";
			int len = getBlobLength(张量句柄);
			if (len != 0)
			{
				float* permute_fc = new float[len];
				try
				{
					cpyBlobData(permute_fc, 张量句柄);

					for (int i = 1; i < time_step; i++)
					{
						o = argmax(permute_fc, (i - 1) * num_output, i * num_output);

						if (o != 空白标签索引 && prev != o && o > -1 && o < num_output)
						{
							rt += labelMap[o];
						}

						prev = o;
					}
				}
				catch (...)
				{
				}
				delete[] permute_fc;
				string strS = findData.cFileName;
				strS = strS.substr(0, strS.find('_', 0));
				string s;
				if (strS == rt)
				{
					正确++;
					s = "正确";
				}
				else
				{
					错误++;
					s = "错误";
					printf("识别的结果是：%s\t%s\t正确答案=%s\n", rt.c_str(), s.c_str(), strS.c_str());
				}
			}
			releaseBlobData(张量句柄);
		}
	} while (FindNextFileA(hFindFine, &findData));
	printf("正确=%d个，错误=%d个，正确率=%f", 正确, 错误, (double)正确 / (错误 + 正确));
	//关闭文件搜索句柄  
	FindClose(hFindFine);
}

void ss(int avgc, char **avgv)
{
	try
	{
		if (!GetProtoParam2(avgv[1]))
		{
			printf("获取proto参数失败\n");
			return;
		}
		printf("num_output = %d, time_step = %d\n", num_output, time_step);
		disableErrorOutput();
		Classifier* classifierHandle = createClassifier(avgv[1], avgv[2]);// , 1, 0, 0, 0, 0, 16);
		loadCodeMap(avgv[3], labelMap);
		/*vector<char> data = readFile(avgv[4]);
		if (data.empty()){
		printf("文件不存在么？\n");
		releaseTaskPool(classifierHandle);
		return;
		}*/
		FindFile2(avgv[4], classifierHandle);
		releaseClassifier(classifierHandle);
	}
	catch (...)
	{
		printf("异常\n");
	}
}
void main(int avgc, char** avgv){
	//return ss(avgc, avgv);
	//禁止caffe输出信息
	if (avgc < 5)return;
	try
	{
		if (!GetProtoParam2(avgv[1]))
		{
			printf("获取proto参数失败\n");
			return;
		}
		printf("num_output = %d, time_step = %d\n", num_output, time_step);
		disableErrorOutput();
		classifierHandle = createTaskPool(avgv[1], avgv[2], 1, 0, 0, 0, 0, MAX_THREAD);
		loadCodeMap(avgv[3], labelMap);
		/*vector<char> data = readFile(avgv[4]);
		if (data.empty()){
			printf("文件不存在么？\n");
			releaseTaskPool(classifierHandle);
			return;
		}*/
		strPath = avgv[4];
		FindFile();
		releaseTaskPool(classifierHandle);
	}
	catch (...)
	{
		printf("异常\n");
	}
}