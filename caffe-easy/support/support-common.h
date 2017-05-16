#pragma once

#ifndef BuildStaticLib
#include "import-caffe-staticlib.h"
#include "import-opencv-staticlib.h"

#ifndef CPU_ONLY
#include "import-cuda-lib.h"
#endif
#endif

#ifdef __cplusplus
#define DllImport __declspec(dllimport)
#define DllExport __declspec(dllexport)
#else
#define DllImport
#define DllExport
#endif

#ifdef BuildDLL
#ifdef ExportDLL
#define Caffe_API DllExport
#else
#define Caffe_API DllImport
#endif
#else
#define Caffe_API
#endif

#ifdef __cplusplus
template<typename Dtype>
class WPtr{

	template<typename T>
	struct ptrInfo{
		T ptr;
		int refCount;

		ptrInfo(T p) :ptr(p), refCount(1){}
		void addRef(){ refCount++; }
		bool releaseRef(){ return --refCount <= 0; }
	};

public:
	WPtr() :ptr(0){};
	WPtr(Dtype p){
		ptr = new ptrInfo<Dtype>(p);
	}
	WPtr(const WPtr& other){
		operator=(other);
	}
	~WPtr(){
		releaseRef();
	}

	void release(Dtype ptr);

	Dtype operator->(){
		return get();
	}

	operator Dtype(){
		return get();
	}

	WPtr& operator=(const WPtr& other){
		releaseRef();

		this->ptr = other.ptr;
		addRef();
		return *this;
	}

	Dtype get(){
		if (this->ptr)
			return ptr->ptr;
		return 0;
	}

	void addRef(){
		if (this->ptr)
			this->ptr->addRef();
	}

	void releaseRef(){
		if (this->ptr && this->ptr->releaseRef()){
			release(this->ptr->ptr);
			delete ptr;
			ptr = 0;
		}
	}

private:
	ptrInfo<Dtype>* ptr;
};

template<typename Dtype>
inline void WPtr<Dtype>::release(Dtype p){
	if (p) delete p;
}
#endif