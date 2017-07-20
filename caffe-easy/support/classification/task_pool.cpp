
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


#define operType_Softmax			1
#define operType_Forward			2

void swapCache(TaskPool* pool){
	pool->recNum = pool->job_cursor;
	if (pool->recNum > 0){
		EnterCriticalSection(&pool->jobCS);
		pool->recNum = pool->job_cursor;
		std::swap(pool->cacheImgs, pool->recImgs);
		std::swap(pool->cacheBlobNames, pool->blobNames);
		std::swap(pool->cacheTop_n, pool->top_n);
		std::swap(pool->cacheOperType, pool->operType);
		std::swap(pool->cacheSemaphoreGetResult, pool->semaphoreGetResult);
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

	//vector<Mat> ims;
	while (pool->flag_run){
		swapCache(pool);
		//printf("recnum = %d\n", pool->recNum);
		if (pool->recNum > 0){
			//ims.clear();
			//for (int i = 0; i < pool->recNum; ++i)
			//	ims.push_back(((Mat*)pool->recImgs)[i]);

			//MultiSoftmaxResult* multi = pool->model->predictSoftmax((Mat*)pool->recImgs, pool->recNum, pool->top_n[0]);
			//printf("pool->recNum = %d\n", pool->recNum);
			pool->model->reshape(pool->recNum, -1, -1);
			pool->model->forward((Mat*)pool->recImgs, pool->recNum);

			for (int i = 0; i < pool->recNum; ++i){
				int type = pool->operType[i];
				switch (type){
				case operType_Forward:
					pool->recBlobs[i] = pool->model->getBlobData(i, (const char*)pool->blobNames[i]);
					break;
				case operType_Softmax:
					pool->recResult[i] = pool->model->getSoftmaxResult(i, pool->top_n[i]);
					break;
				}
			}
			//MultiSoftmaxResult* multi = predictMultiSoftmax(pool->model, (const void**)pool->recImgs, (int*)pool->recLengths, pool->recNum, pool->top_n[0]);
			//memcpy(pool->recResult, multi->list, sizeof(SoftmaxResult*)* multi->count);
			//delete [] multi->list;
			//delete multi;

			for (int i = 0; i < pool->recNum; ++i)
				ReleaseSemaphore(pool->semaphoreGetResult[i], 1, 0);

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

	pool->recImgs = new volatile Mat[batch_size];
	pool->top_n = new volatile int[batch_size];
	pool->blobNames = new volatile char*[batch_size];
	pool->operType = new int[batch_size];

	pool->cacheImgs = new volatile Mat[batch_size];
	pool->cacheTop_n = new volatile int[batch_size];
	pool->cacheBlobNames = new volatile char*[batch_size];
	pool->cacheOperType = new int[batch_size];

	pool->semaphoreWait = CreateSemaphoreA(0, batch_size, batch_size, 0);
	pool->recResult = new volatile SoftmaxResult*[batch_size];
	pool->recBlobs = new volatile BlobData*[batch_size];

	//pool->semaphoreGetResult = CreateSemaphoreA(0, 0, batch_size, 0);
	pool->semaphoreGetResultFinish = CreateSemaphoreA(0, 0, batch_size, 0);
	pool->gpu_id = gpu_id;
	pool->flag_exit = false;
	pool->flag_run = false;
	pool->semaphoreGetResult = new volatile HANDLE[batch_size];
	pool->cacheSemaphoreGetResult = new volatile HANDLE[batch_size];
	for (int i = 0; i < batch_size; ++i){
		pool->semaphoreGetResult[i] = CreateSemaphoreA(0, 0, 1, 0);
		pool->cacheSemaphoreGetResult[i] = CreateSemaphoreA(0, 0, 1, 0);
	}

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
	ReleaseSemaphore(pool->semaphoreGetResultFinish, pool->count_worker, 0);
	while (!pool->flag_exit)
		Sleep(10);

	CloseHandle(pool->semaphoreWait);
	for (int i = 0; i < pool->count_worker; ++i){
		CloseHandle(pool->semaphoreGetResult[i]);
		CloseHandle(pool->cacheSemaphoreGetResult[i]);
	}

	CloseHandle(pool->semaphoreGetResultFinish);
	DeleteCriticalSection(&pool->jobCS);
	
	delete[] pool->recImgs;
	delete[] pool->top_n;
	delete[] pool->blobNames;
	delete[] pool->operType;

	delete[] pool->cacheImgs;
	delete[] pool->cacheTop_n;
	delete[] pool->cacheBlobNames;
	delete[] pool->cacheOperType;

	delete[] pool->recResult;
	delete[] pool->semaphoreGetResult;
	delete[] pool->cacheSemaphoreGetResult;
	delete[] pool->recBlobs;
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
	return predictSoftmaxByTaskPool2(pool, &im, top_n);
}

Caffe_API SoftmaxResult* __stdcall predictSoftmaxAnyByTaskPool(TaskPool* pool, const float* data, const int* dims, int top_n){
	if (pool == 0 || data == 0 || dims == 0 || top_n < 1) return 0;
	if (!pool->flag_run) return 0;

	Mat im(dims[1], dims[2], CV_32FC(dims[0]), (uchar*)data);
	if (im.empty()) return 0;
	return predictSoftmaxByTaskPool2(pool, &im, top_n);
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
	HANDLE semaphore = pool->cacheSemaphoreGetResult[cursor];
	((Mat*)pool->cacheImgs)[cursor] = im;
	pool->cacheTop_n[cursor] = top_n;
	pool->cacheOperType[cursor] = operType_Softmax;
	pool->job_cursor++;
	LeaveCriticalSection(&pool->jobCS);

	WaitForSingleObject(semaphore, -1);
	volatile SoftmaxResult* result = pool->recResult[cursor];
	ReleaseSemaphore(pool->semaphoreGetResultFinish, 1, 0);
	return (SoftmaxResult*)result;
}

Caffe_API BlobData* __stdcall forwardByTaskPool(TaskPool* pool, const void* img, int len, const char* blob_name){
	if (pool == 0 || img == 0 || len < 1) return 0;
	if (!pool->flag_run) return 0;

	Mat im;
	try{
		im = cv::imdecode(Mat(1, len, CV_8U, (uchar*)img), pool->model->input_channels() == 3 ? 1 : 0);
	}
	catch (...){
		return 0;
	}

	if (im.empty()) return 0;
	return forwardByTaskPool2(pool, &im, blob_name);
}

Caffe_API BlobData* __stdcall forwardByTaskPool2(TaskPool* pool, const Image* img, const char* blob_name){
	if (pool == 0 || img == 0) return 0;
	if (!pool->flag_run) return 0;

	if (img->empty()) return 0;

	Mat im;
	img->copyTo(im);
	if (im.empty()) return 0;

	WaitForSingleObject(pool->semaphoreWait, -1);
	if (!pool->flag_run) return 0;

	EnterCriticalSection(&pool->jobCS);
	int cursor = pool->job_cursor;
	HANDLE semaphore = pool->cacheSemaphoreGetResult[cursor];
	((Mat*)pool->cacheImgs)[cursor] = im;
	pool->cacheTop_n[cursor] = -1;
	pool->cacheOperType[cursor] = operType_Forward;
	pool->cacheBlobNames[cursor] = (char*)blob_name;
	pool->job_cursor++;
	LeaveCriticalSection(&pool->jobCS);

	WaitForSingleObject(semaphore, -1);
	volatile BlobData* result = pool->recBlobs[cursor];
	ReleaseSemaphore(pool->semaphoreGetResultFinish, 1, 0);
	return (BlobData*)result;
}
