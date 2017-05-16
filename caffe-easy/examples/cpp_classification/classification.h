#pragma once

#ifdef __cplusplus
#include <opencv/cv.h>
#include <string>
#include <vector>
#endif

#ifdef __cplusplus
#define DllImport __declspec(dllimport)
#define DllExport __declspec(dllexport)
#else
#define DllImport
#define DllExport
#endif

#ifdef Caffe_BuildDLL
#define Caffe_API DllExport
#else
#define Caffe_API DllImport
#endif

#ifdef __cplusplus
class CaffeClassifier;
#else
typedef void* CaffeClassifier;
#endif

#ifdef __cplusplus
extern "C"{
#endif
	struct SoftmaxData{
		int label;
		float conf;
	};

	struct SoftmaxLayerOutput{
		int count;
		SoftmaxData* result;
	};

	struct SoftmaxResult{
		int count;
		SoftmaxLayerOutput* list;
	};

	struct FeatureResult{
		int count;
		float* list;
	};

	Caffe_API void  __stdcall releaseFeatureResult(FeatureResult* ptr);
	Caffe_API void  __stdcall releaseSoftmaxResult(SoftmaxResult* ptr);

	Caffe_API CaffeClassifier* __stdcall createClassifier(
		const char* prototxt_file,
		const char* caffemodel_file,
		float scale_raw = 1,
		const char* mean_file = 0,
		int num_means = 0,
		float* means = 0);

	Caffe_API void __stdcall releaseClassifier(CaffeClassifier* classifier);
	Caffe_API SoftmaxResult* __stdcall predictSoftmax(CaffeClassifier*classifier, const void* img, int len, int top_n = 5);
	Caffe_API FeatureResult* __stdcall extfeature(CaffeClassifier*classifier, const void* img, int len, const char* feature_name);

	//获取特征的长度
	Caffe_API int __stdcall getFeatureLength(FeatureResult* feature);

	//将特征复制到缓存区
	Caffe_API void __stdcall cpyFeature(void* buffer, FeatureResult* feature);

	//获取输出层的个数
	Caffe_API int __stdcall getNumOutlayers(SoftmaxResult* result);

	//获取层里面的数据个数
	Caffe_API int __stdcall getLayerNumData(SoftmaxLayerOutput* layer);

	//获取结果的label
	Caffe_API int __stdcall getResultLabel(SoftmaxResult* result, int layer, int num);

	//获取结果的置信度
	Caffe_API float __stdcall getResultConf(SoftmaxResult* result, int layer, int num);

	//多标签就是多个输出层，每个层取softmax，注意buf的个数是getNumOutlayers得到的数目一致
	Caffe_API void __stdcall getMultiLabel(SoftmaxResult* result, int* buf);

	//获取第0个输出的label
	Caffe_API int __stdcall getSingleLabel(SoftmaxResult* result);

	//获取第0个输出的置信度
	Caffe_API float __stdcall getSingleConf(SoftmaxResult* result);

	//获取最后发生的错误，没有错误返回0
	Caffe_API const char* __stdcall getLastErrorMessage();

	Caffe_API void __stdcall enablePrintErrorToConsole();

	Caffe_API void __stdcall disableErrorOutput();
#ifdef __cplusplus
}; 
#endif

#ifdef __cplusplus
class Caffe_API CaffeClassifier {
public:
	CaffeClassifier(const char* prototxt_file,
		const char* caffemodel_file,
		float scale_raw = 1,
		const char* mean_file = 0,
		int num_means = 0,
		float* means = 0);

	~CaffeClassifier();
public:
	SoftmaxResult* predictSoftmax(const cv::Mat& img, int top_n = 5);
	FeatureResult* extfeature(const cv::Mat& img, const char* feature_name);

private:
	void* native_;
};
#endif