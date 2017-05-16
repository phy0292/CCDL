
#include <caffe/caffe.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "classification.h"
#include "classification-c.h"
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;
using namespace std;
using namespace cv;

static char g_last_error[4096] = { 0 };
static bool g_is_print_err_to_console = false;
static volatile DecipherCallback decipherCallback = 0;

#define ThisNet ((Net<float>*)net_)

#define Version		1.0
#define	snprintf	_snprintf
#define vsnprintf	_vsnprintf
#define errBegin	try{
#define errEnd(...)	}catch (const exception& e){setError(__FILE__, __LINE__,__FUNCTION__,"%s", e.what());}catch(...){setError(__FILE__, __LINE__,__FUNCTION__,"unknow error");} return __VA_ARGS__;
#define errmsg(...) setError(__FILE__, __LINE__,__FUNCTION__, __VA_ARGS__)

Caffe_API void __stdcall disableErrorOutput(){
	google::InitGoogleLogging("aa");
}

Caffe_API float __stdcall getVersion(DecipherCallback callback){
	decipherCallback = callback;
	return Version;
}

void setError(const char* file, int line, const char* function, const char* fmt, ...){
	va_list va;
	va_start(va, fmt);

	char err[3000];
	vsnprintf(err, sizeof(err), fmt, va);
	snprintf(g_last_error, sizeof(g_last_error), "file[%d]: %s\nfunction: %s\nerror: %s", line, file, function, err);

	if (g_is_print_err_to_console)
		printf("%s\n", g_last_error);
}

void __stdcall freeBlobData(BlobData** pptr){
	if (pptr){
		BlobData* ptr = *pptr;
		if (ptr){
			if (ptr->list){
				delete ptr->list;
				ptr->list = 0;
			}
			delete ptr;
		}
		*pptr = 0;
	}
}

void __stdcall freeSoftmaxResult(SoftmaxResult** pptr){
	if (pptr){
		SoftmaxResult* ptr = *pptr;
		if (ptr){
			if (ptr->list){
				for (int i = 0; i < ptr->count; ++i){
					delete ptr->list[i].result;
					ptr->list[i].result = 0;
				}

				delete ptr->list;
				ptr->list = 0;
			}
			delete ptr;
		}
		*pptr = 0;
	}
}

Caffe_API void  __stdcall releaseBlobData(BlobData* ptr){
	freeBlobData(&ptr);
}

Caffe_API void  __stdcall releaseSoftmaxResult(SoftmaxResult* ptr){
	freeSoftmaxResult(&ptr);
}

Caffe_API Classifier* __stdcall createClassifier(
	const char* prototxt_file,
	const char* caffemodel_file,
	float scale_raw,
	const char* mean_file,
	int num_means,
	float* means,
	int gpu_id){

	errBegin;
	return new Classifier(prototxt_file, caffemodel_file, scale_raw, mean_file, num_means, means, gpu_id);
	errEnd(0);
}

Caffe_API Classifier* __stdcall createClassifierByData(
	const void* prototxt_data,
	int prototxt_data_length,
	const void* caffemodel_data,
	int caffemodel_data_length,
	float scale_raw,
	const char* mean_file,
	int num_means,
	float* means,
	int gpu_id){

	void* ret_pro = 0;
	void* ret_model = 0;

	errBegin;
		void* _pro_data = (void*)prototxt_data;
		int _pro_len = prototxt_data_length;
		void* _model_data = (void*)caffemodel_data;
		int _model_len = caffemodel_data_length;

		if (decipherCallback != 0){
			ret_pro = decipherCallback(event_decipher, type_prototxt, _pro_data, _pro_len);
			if (ret_pro == 0){
				errmsg("fail to decode.");
				return 0;
			}

			memcpy(&_pro_len, ret_pro, sizeof(_pro_len));
			if (_pro_len == 0){
				errmsg("fail pro len");
				decipherCallback(event_free, type_prototxt, ret_pro, 0); ret_pro = 0;
				return 0;
			}
			_pro_data = (char*)ret_pro + sizeof(_pro_len);

			ret_model = decipherCallback(event_decipher, type_caffemodel, _model_data, _model_len);
			if (ret_model == 0){
				errmsg("fail to decodem.");
				decipherCallback(event_free, type_prototxt, ret_pro, 0); ret_pro = 0;
				return 0;
			}

			memcpy(&_model_len, ret_model, sizeof(_model_len));
			if (_model_len == 0){
				errmsg("fail mdl len");
				decipherCallback(event_free, type_prototxt, ret_pro, 0); ret_pro = 0;
				decipherCallback(event_free, type_caffemodel, ret_model, 0); ret_model = 0;
				return 0;
			}
			_model_data = (char*)ret_model + sizeof(_model_len);
		}

		Classifier* out = new Classifier(
			_pro_data, _pro_len, _model_data, _model_len, 
			scale_raw, mean_file, num_means, means, gpu_id);

		if (ret_pro != 0)
			decipherCallback(event_free, type_prototxt, ret_pro, 0); ret_pro = 0;

		if (ret_model != 0)
			decipherCallback(event_free, type_caffemodel, ret_model, 0); ret_model = 0;
		return out;
	}
	catch (const exception& e){ setError(__FILE__, __LINE__, __FUNCTION__, "%s", e.what()); }
	catch (...){ setError(__FILE__, __LINE__, __FUNCTION__, "unknow error"); }

	if (ret_pro != 0)
		decipherCallback(event_free, type_prototxt, ret_pro, 0); ret_pro = 0;

	if (ret_model != 0)
		decipherCallback(event_free, type_caffemodel, ret_model, 0); ret_model = 0;
	return 0;
}

Caffe_API void __stdcall releaseClassifier(Classifier* classifier){
	if (classifier != 0)
		delete classifier;
}

Caffe_API SoftmaxResult* __stdcall predictSoftmax(Classifier* Classifier, const void* img, int len, int top_n){
	errBegin;
	if (Classifier == 0 || len < 1 || img == 0) return 0;

	Mat im;
	try{
		im = imdecode(Mat(1, len, CV_8U, (uchar*)img), 1);
	}catch (...){}
	if (im.empty()) return 0;

	return Classifier->predictSoftmax(im, top_n);
	errEnd(0);
}

Caffe_API BlobData* __stdcall extfeature(Classifier* Classifier, const void* img, int len, const char* feature_name){
	errBegin;
	if (Classifier == 0 || len < 1 || img == 0) return 0;

	Mat im;
	
	try{
		im = imdecode(Mat(1, len, CV_8U, (uchar*)img), 1);
	}catch (...){}
	if (im.empty()) return 0;

	return Classifier->extfeature(im, feature_name);
	errEnd(0);
}

Caffe_API int __stdcall getFeatureLength(BlobData* feature){
	if (!feature) return 0;
	return feature->count;
}

Caffe_API void __stdcall cpyFeature(void* buffer, BlobData* feature){
	if (!feature) return;
	memcpy(buffer, feature->list, sizeof(feature->list[0])*feature->count);
}

Caffe_API int __stdcall getNumOutlayers(SoftmaxResult* result){
	if (!result) return 0;
	return result->count;
}

Caffe_API int __stdcall getLayerNumData(SoftmaxLayerOutput* layer){
	if (!layer) return 0;
	return layer->count;
}

Caffe_API int __stdcall getResultLabel(SoftmaxResult* result, int layer, int num){
	if (!result || !result->list || layer >= result->count || layer < 0 || num >= result->list[layer].count || num < 0 || !result->list[layer].result) return -1;
	return result->list[layer].result[num].label;
}

Caffe_API float __stdcall getResultConf(SoftmaxResult* result, int layer, int num){
	if (!result || !result->list || layer >= result->count || layer < 0 || num >= result->list[layer].count || num < 0 || !result->list[layer].result) return -1;
	return result->list[layer].result[num].conf;
}

//多标签就是多个输出层，每个层取softmax
Caffe_API void __stdcall getMultiLabel(SoftmaxResult* result, int* buf){
	if (!result || result->count == 0 || !result->list) return;

	for (int i = 0; i < result->count; ++i){
		if (!result->list[i].result || result->list[i].count == 0){
			buf[i] = -1;
			continue;
		}

		buf[i] = result->list[i].result[0].label;
	}
}

//获取第0个输出的label
Caffe_API int __stdcall getSingleLabel(SoftmaxResult* result){
	if (!result) return -1;
	if (result->count == 0 || !result->list || result->list[0].count == 0 || !result->list[0].result) return -1;
	return result->list[0].result[0].label;
}

//获取第0个输出的置信度
Caffe_API float __stdcall getSingleConf(SoftmaxResult* result){
	if (!result) return -1;
	if (result->count == 0 || !result->list || result->list[0].count == 0 || !result->list[0].result) return -1;
	return result->list[0].result[0].conf;
}

//获取最后发生的错误，没有错误返回0
Caffe_API const char* __stdcall getLastErrorMessage(){
	return g_last_error;
}

Caffe_API void __stdcall enablePrintErrorToConsole(){
	g_is_print_err_to_console = true;
}

bool fileExists(const char* file){
	FILE* f = fopen(file, "rb");
	if (f != 0){
		fclose(f);
		return true;
	}
	return false;
}

void setGPU(int gpu_id){
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	if (gpu_id < 0) {
		Caffe::set_mode(Caffe::CPU);
	}
	else {
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(gpu_id);
	}
#endif
}

int Classifier::getOutputBlobCount(){
	return ThisNet->num_outputs();
}

BlobData* Classifier::getOutputBlob(int index){
	if (index < 0 || index >= ThisNet->num_outputs()){
		cout << "layer index out of range.";
		return 0;
	}

	return getBlobDataByRawBlob(ThisNet->output_blobs()[index]);
}

BlobData* Classifier::getBlobDataByRawBlob(void* _blob){
	Blob<float>* blob = (Blob<float>*)_blob;
	const float* begin = blob->cpu_data();
	BlobData* fresult = new BlobData();
	fresult->count = blob->count();
	fresult->list = new float[fresult->count];
	fresult->channels = blob->channels();
	fresult->num = blob->num();
	fresult->height = blob->height();
	fresult->width = blob->width();
	memcpy(fresult->list, begin, sizeof(float)*fresult->count);
	return fresult;
}

BlobData* Classifier::getBlobData(const char* blob_name){
	if (blob_name == 0 || *blob_name == 0)
		return getOutputBlob(0);

	if (!ThisNet->has_blob(blob_name)){
		printf("no blob: %s\n", blob_name);
		return 0;
	}
	return getBlobDataByRawBlob(ThisNet->blob_by_name(blob_name).get());
}

Classifier::Classifier(const void* prototxt_data,
	int prototxt_data_length,
	const void* caffemodel_data,
	int caffemodel_data_length,
	float scale_raw,
	const char* mean_file,
	int num_means,
	float* means,
	int gpu_id){

	this->scale_raw = scale_raw;
	setGPU(gpu_id);

	/* Load the network. */
	net_ = new Net<float>(prototxt_data, prototxt_data_length, TEST);
	ThisNet->CopyTrainedLayersFromData(caffemodel_data, caffemodel_data_length);

	CHECK_EQ(ThisNet->num_inputs(), 1) << "Network should have exactly one input.";

	
	if (ThisNet->num_inputs() > 0){
		Blob<float>* input_layer = ThisNet->input_blobs()[0];
		num_channels_ = input_layer->channels();
		CHECK(num_channels_ == 3 || num_channels_ == 1)
			<< "Input layer should have 1 or 3 channels.";
		input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
	}

	/* Load the binaryproto mean file. */
	if (mean_file != 0 && fileExists(mean_file))
		SetMean(mean_file);
	else{
		if (num_means > 0 && means != 0){
			Scalar mean_scal;
			for (int i = 0; i < num_means; ++i)
				mean_scal[i] = means[i];
			mean_ = cv::Mat(input_geometry_, CV_32FC(num_means), mean_scal);
		}
	}
}

Classifier::Classifier(const char* prototxt_file,
	const char* caffemodel_file,
	float scale_raw,
	const char* mean_file,
	int num_means,
	float* means,
	int gpu_id) {

	this->scale_raw = scale_raw;
	setGPU(gpu_id);
  
  /* Load the network. */
  net_ = new Net<float>(prototxt_file, TEST);
  ThisNet->CopyTrainedLayersFrom(caffemodel_file);

  CHECK_EQ(ThisNet->num_inputs(), 1) << "Network should have exactly one input.";

  
  if (ThisNet->num_inputs() > 0){
	  Blob<float>* input_layer = ThisNet->input_blobs()[0];
	  num_channels_ = input_layer->channels();
	  CHECK(num_channels_ == 3 || num_channels_ == 1)
	    << "Input layer should have 1 or 3 channels.";
	  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  }
  

  /* Load the binaryproto mean file. */
  if (mean_file != 0 && fileExists(mean_file))
	SetMean(mean_file);
  else{
	  if (num_means > 0 && means != 0){
		  Scalar mean_scal;
		  for (int i = 0; i < num_means; ++i)
			  mean_scal[i] = means[i];
		  mean_ = cv::Mat(input_geometry_, CV_32FC(num_means), mean_scal);
	  }
  }
}

Classifier::~Classifier(){
	delete ThisNet;
	net_ = 0;
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

BlobData* Classifier::extfeature(const cv::Mat& img, const char* feature_name){
	Blob<float>* input_layer = ThisNet->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	ThisNet->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	ThisNet->Forward();
	return getBlobData(feature_name);
}

/* Return the top N predictions. */
SoftmaxResult* Classifier::predictSoftmax(const cv::Mat& img, int top_n) {
  std::vector<std::vector<float>> output;
  Predict(img, output);
  SoftmaxResult* result = new SoftmaxResult();
  result->count = output.size();
  result->list = new SoftmaxLayerOutput[result->count];

  int N;
  for (int i = 0; i < result->count; ++i){
	  N = top_n;
	  N = N > output[i].size() ? output[i].size() : N;
	  result->list[i].result = new SoftmaxData[N];
	  result->list[i].count = N;

	  std::vector<int> maxN = Argmax(output[i], N);
	  for (int k = 0; k < N; ++k) {
		  int idx = maxN[k];
		  result->list[i].result[k].label = idx;
		  result->list[i].result[k].conf = output[i][idx];
	  }
  }
  return result;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const char* mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

void Classifier::Predict(const cv::Mat& img, std::vector<std::vector<float> >& out) {
  Blob<float>* input_layer = ThisNet->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  ThisNet->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  ThisNet->Forward();

  /* Copy the output layer to a std::vector */
  out.resize(ThisNet->output_blobs().size());
  for (int i = 0; i < ThisNet->output_blobs().size(); ++i){
	  Blob<float>* output_layer = ThisNet->output_blobs()[i];
	  const float* begin = output_layer->cpu_data();
	  const float* end = begin + output_layer->channels();
	  out[i] = std::vector<float>(begin, end);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = ThisNet->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  if (!mean_.empty())
	  cv::subtract(sample_float, mean_, sample_normalized);
  else
	  sample_float.copyTo(sample_normalized);

  if (this->scale_raw != 0)
	sample_normalized *= this->scale_raw;
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == ThisNet->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int Classifier::input_num(int index){
	return ThisNet->input_blobs()[index]->num();
}

int Classifier::input_channels(int index){
	return ThisNet->input_blobs()[index]->channels();
}

int Classifier::input_width(int index){
	return ThisNet->input_blobs()[index]->width();
}

int Classifier::input_height(int index){
	return ThisNet->input_blobs()[index]->height();
}
