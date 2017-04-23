#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "classification.h"
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/shared_ptr.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace std;
using namespace cv;

static char g_last_error[4096] = { 0 };
static bool g_is_print_err_to_console = false;

#define	snprintf	_snprintf
#define vsnprintf	_vsnprintf
#define errBegin	try{
#define errEnd(...)	}catch (const exception& e){setError(__FILE__, __LINE__,__FUNCTION__,"%s", e.what());}catch(...){setError(__FILE__, __LINE__,__FUNCTION__,"unknow error");} return __VA_ARGS__;

static void ProcessCore(const char* data, int size){
	printf("hope: %s\n", string(data, size).c_str());
}

Caffe_API void __stdcall disableErrorOutput(){
	google::InitGoogleLogging("aa");
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

void freeFeatureResult(FeatureResult** pptr){
	if (pptr){
		FeatureResult* ptr = *pptr;
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

void freeSoftmaxResult(SoftmaxResult** pptr){
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

Caffe_API void  __stdcall releaseFeatureResult(FeatureResult* ptr){
	freeFeatureResult(&ptr);
}

Caffe_API void  __stdcall releaseSoftmaxResult(SoftmaxResult* ptr){
	freeSoftmaxResult(&ptr);
}

Caffe_API CaffeClassifier* __stdcall createClassifier(
	const char* prototxt_file,
	const char* caffemodel_file,
	float scale_raw,
	const char* mean_file,
	int num_means,
	float* means){

	errBegin;
	return new CaffeClassifier(prototxt_file, caffemodel_file, scale_raw, mean_file, num_means, means);
	errEnd(0);
}

Caffe_API void __stdcall releaseClassifier(CaffeClassifier* classifier){
	if (classifier != 0)
		delete classifier;
}

Caffe_API SoftmaxResult* __stdcall predictSoftmax(CaffeClassifier* classifier, const void* img, int len, int top_n){
	errBegin;
	if (classifier == 0 || len < 1 || img == 0) return 0;

	Mat im = imdecode(Mat(1, len, CV_8U, (uchar*)img), 1);
	if (im.empty()) return 0;

	return classifier->predictSoftmax(im, top_n);
	errEnd(0);
}

Caffe_API FeatureResult* __stdcall extfeature(CaffeClassifier* classifier, const void* img, int len, const char* feature_name){
	errBegin;
	if (classifier == 0 || len < 1 || img == 0) return 0;

	Mat im = imdecode(Mat(1, len, CV_8U, (uchar*)img), 1);
	if (im.empty()) return 0;

	return classifier->extfeature(im, feature_name);
	errEnd(0);
}

Caffe_API int __stdcall getFeatureLength(FeatureResult* feature){
	if (!feature) return 0;
	return feature->count;
}

Caffe_API void __stdcall cpyFeature(void* buffer, FeatureResult* feature){
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

class Classifier {
public:
	Classifier(const char* model_file,
		const char* trained_file,
		float scale_raw = 1,
		const char* mean_file = "",
		int num_means = 0,
		float* means = 0);

public:
	SoftmaxResult* predictSoftmax(const cv::Mat& img, int top_n = 5);
	FeatureResult* extfeature(const cv::Mat& img, const char* feature_name);
	 
private:
	void SetMean(const string& mean_file);

	void Predict(const cv::Mat& img, std::vector<std::vector<float> >& out);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	boost::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	float scale_raw;
};

bool fileExists(const char* file){
	FILE* f = fopen(file, "rb");
	if (f != 0){
		fclose(f);
		return true;
	}
	return false;
}

Classifier::Classifier(const char* model_file,
	const char* trained_file,
	float scale_raw,
	const char* mean_file,
	int num_means,
	float* means) {

	this->scale_raw = scale_raw;
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

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

FeatureResult* Classifier::extfeature(const cv::Mat& img, const char* feature_name){
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();
	const boost::shared_ptr<Blob<float>> output_layer = 
		feature_name == 0 ? 
		boost::shared_ptr<Blob<float>>(net_->output_blobs()[0]) : 
		net_->blob_by_name(feature_name);

	const float* begin = output_layer->cpu_data();
	//const float* end = begin + output_layer->channels();

	FeatureResult* fresult = new FeatureResult();
	fresult->count = output_layer->count();
	fresult->list = new float[fresult->count];
	memcpy(fresult->list, begin, sizeof(float)*fresult->count);
	return fresult;
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
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

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
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  out.resize(net_->output_blobs().size());
  for (int i = 0; i < net_->output_blobs().size(); ++i){
	  Blob<float>* output_layer = net_->output_blobs()[i];
	  const float* begin = output_layer->cpu_data();
	  const float* end = begin + output_layer->channels();
	  out[i] = std::vector<float>(begin, end);
  }
  //Blob<float>* output_layer = net_->output_blobs()[0];
  //const float* begin = output_layer->cpu_data();
  //const float* end = begin + output_layer->channels();
  //return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

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
  /* Convert the input image to the input image format of the network. */
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
	sample_normalized /= this->scale_raw;
  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  //printf("%d %d %d\n", sample_normalized.rows, sample_normalized.cols, sample_normalized.channels());
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

CaffeClassifier::CaffeClassifier(
	const char* prototxt_file,
	const char* caffemodel_file,
	float scale_raw,
	const char* mean_file,
	int num_means,
	float* means){
	
	errBegin;
	this->native_ = new Classifier(prototxt_file, caffemodel_file, scale_raw, mean_file, num_means, means);
	errEnd();
}

CaffeClassifier::~CaffeClassifier(){
	Classifier* ptr = (Classifier*)this->native_;
	if (ptr != 0)
		delete ptr;
}

SoftmaxResult* CaffeClassifier::predictSoftmax(const cv::Mat& img, int top_n){
	Classifier* ptr = (Classifier*)this->native_;

	errBegin;
	return ptr->predictSoftmax(img, top_n);
	errEnd(0);
}

FeatureResult* CaffeClassifier::extfeature(const cv::Mat& img, const char* feature_name){
	Classifier* ptr = (Classifier*)this->native_;
	errBegin;
	return ptr->extfeature(img, feature_name);
	errEnd(0);
}