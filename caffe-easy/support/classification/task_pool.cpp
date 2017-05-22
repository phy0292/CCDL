
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

#include <Windows.h>
#include <process.h>
using namespace caffe;
using namespace std;
using namespace cv;

void swapCache(TaskPool* pool){
	pool->recNum = pool->job_cursor;
	if (pool->recNum > 0){
		EnterCriticalSection(&pool->jobCS);
		pool->recNum = pool->job_cursor;
		std::swap(pool->cacheImgs, pool->recImgs);
		pool->job_cursor = 0;
		LeaveCriticalSection(&pool->jobCS);
		ReleaseSemaphore(pool->semaphoreWait, pool->recNum, 0);
	}

	if (pool->recNum == 0)
		Sleep(1);
}

extern void setGPU(int gpu_id);
void poolThread(void* param){
	TaskPool* pool = (TaskPool*)param;
	pool->flag_run = true;

	// GPU是线程上下文相关的
	setGPU(pool->gpu_id);

	vector<Mat> ims;
	while (pool->flag_run){
		swapCache(pool);
		if (pool->recNum > 0){
			ims.clear();
			for (int i = 0; i < pool->recNum; ++i)
				ims.push_back(((Mat*)pool->recImgs)[i]);

			MultiSoftmaxResult* multi = pool->model->predictSoftmax(ims, pool->top_n[0]);
			//MultiSoftmaxResult* multi = predictMultiSoftmax(pool->model, (const void**)pool->recImgs, (int*)pool->recLengths, pool->recNum, pool->top_n[0]);
			memcpy(pool->recResult, multi->list, sizeof(SoftmaxResult*)* multi->count);
			delete [] multi->list;
			delete multi;

			ReleaseSemaphore(pool->semaphoreGetResult, pool->recNum, 0);
			for (int i = 0; i < pool->recNum; ++i)
				WaitForSingleObject(pool->semaphoreGetResultFinish, -1);
		}
	}
	pool->flag_run = false;
	pool->flag_exit = true;
}

TaskPool* buildPool(Classifier* model, int gpu_id, int batch_size){
	batch_size = batch_size < 1 ? 1 : batch_size;

	TaskPool* pool = new TaskPool();
	memset(pool, 0, sizeof(*pool));
	pool->model = model;
	pool->count_worker = batch_size;
	pool->cacheImgs = new volatile Mat[batch_size];
	pool->top_n = new volatile int[batch_size];
	pool->recImgs = new volatile Mat[batch_size];
	pool->recResult = new volatile SoftmaxResult*[batch_size];
	pool->semaphoreWait = CreateSemaphoreA(0, batch_size, batch_size, 0);
	pool->semaphoreGetResult = CreateSemaphoreA(0, 0, batch_size, 0);
	pool->semaphoreGetResultFinish = CreateSemaphoreA(0, 0, batch_size, 0);
	pool->gpu_id = gpu_id;
	pool->flag_exit = false;
	pool->flag_run = false;
	InitializeCriticalSection(&pool->jobCS);
	_beginthread(poolThread, 0, pool);
	return pool;
}

Caffe_API TaskPool* __stdcall createTaskPool(
	const char* prototxt_file,
	const char* caffemodel_file,
	float scale_raw,
	const char* mean_file,
	int num_means,
	float* means,
	int gpu_id,
	int batch_size){

	Classifier* classifier = createClassifier(prototxt_file, caffemodel_file, scale_raw, mean_file, num_means, means, -1);
	if (classifier == 0) return 0;
	return buildPool(classifier, gpu_id, batch_size);
}

Caffe_API TaskPool* __stdcall createTaskPoolByData(
	const void* prototxt_data,
	int prototxt_data_length,
	const void* caffemodel_data,
	int caffemodel_data_length,
	float scale_raw,
	const char* mean_file,
	int num_means,
	float* means,
	int gpu_id,
	int batch_size){

	Classifier* classifier = createClassifierByData(prototxt_data, prototxt_data_length, caffemodel_data, caffemodel_data_length, scale_raw, mean_file, num_means, means, -1);
	if (classifier == 0) return 0;
	return buildPool(classifier, gpu_id, batch_size);
}

Caffe_API void __stdcall releaseTaskPool(TaskPool* pool){
	if (pool == 0) return;

	pool->flag_run = false;
	ReleaseSemaphore(pool->semaphoreWait, pool->count_worker, 0);
	ReleaseSemaphore(pool->semaphoreGetResult, pool->count_worker, 0);
	ReleaseSemaphore(pool->semaphoreGetResultFinish, pool->count_worker, 0);
	while (!pool->flag_exit)
		Sleep(10);

	CloseHandle(pool->semaphoreWait);
	CloseHandle(pool->semaphoreGetResult);
	CloseHandle(pool->semaphoreGetResultFinish);
	DeleteCriticalSection(&pool->jobCS);
	
	delete[] pool->cacheImgs;
	delete[] pool->recImgs;
	delete[] pool->recResult;
	delete[] pool->top_n;
	delete pool;
}

Caffe_API SoftmaxResult* __stdcall predictSoftmaxByTaskPool(TaskPool* pool, const void* img, int len, int top_n){
	
	if (pool == 0 || img == 0 || len < 1 || top_n < 1) return 0;
	if (!pool->flag_run) return 0;

	Mat im;
	try{
		im = cv::imdecode(Mat(1, len, CV_8U, (uchar*)img), pool->model->input_channels() == 3 ? 1 : 0);
	}
	catch (...){
		return 0;
	}

	if (im.empty()) return 0;
	return predictSoftmaxByTaskPool2(pool, &im, 1);
#if 0
	if (pool == 0 || img == 0 || len < 1 || top_n < 1) return 0;
	if (!pool->flag_run) return 0;

	Mat im;
	try{
		im = cv::imdecode(Mat(1, len, CV_8U, (uchar*)img), pool->model->input_channels() == 3 ? 1 : 0);
	}catch(...){
		return 0;
	}

	if (im.empty()) return 0;

	WaitForSingleObject(pool->semaphoreWait, -1);
	if (!pool->flag_run) return 0;

	EnterCriticalSection(&pool->jobCS);
	int cursor = pool->job_cursor;
	((Mat*)pool->cacheImgs)[cursor] = im;
	pool->top_n[cursor] = top_n;
	pool->job_cursor++;
	LeaveCriticalSection(&pool->jobCS);

	WaitForSingleObject(pool->semaphoreGetResult, -1);
	SoftmaxResult* result = (SoftmaxResult*)pool->recResult[cursor];
	ReleaseSemaphore(pool->semaphoreGetResultFinish, 1, 0);
	return result;
#endif
}

//Caffe_API SoftmaxResult* __stdcall predictSoftmaxByTaskPool2(TaskPool* pool, const Image* img, int len, int top_n = 5)
Caffe_API SoftmaxResult* __stdcall predictSoftmaxByTaskPool2(TaskPool* pool, const Image* img, int top_n){
	if (pool == 0 || img == 0 || top_n < 1) return 0;
	if (!pool->flag_run) return 0;

	if (img->empty()) return 0;

	Mat im;
	img->copyTo(im);
	if (im.empty()) return 0;

	WaitForSingleObject(pool->semaphoreWait, -1);
	if (!pool->flag_run) return 0;

	EnterCriticalSection(&pool->jobCS);
	int cursor = pool->job_cursor;
	((Mat*)pool->cacheImgs)[cursor] = im;
	pool->top_n[cursor] = top_n;
	pool->job_cursor++;
	LeaveCriticalSection(&pool->jobCS);

	WaitForSingleObject(pool->semaphoreGetResult, -1);
	SoftmaxResult* result = (SoftmaxResult*)pool->recResult[cursor];
	ReleaseSemaphore(pool->semaphoreGetResultFinish, 1, 0);
	return result;
}