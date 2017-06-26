#ifndef CAFFE_MTCNN_CLS_LAYER_HPP_
#define CAFFE_MTCNN_CLS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Also known as a "fully-connected" layer, computes an inner product
	*        with a set of learned weights, and (optionally) adds biases.
	*
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class MTCNNClsLayer : public Layer<Dtype> {
	public:
		explicit MTCNNClsLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "MTCNNCls"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 2; }
		 
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int N;
		std::vector<int> valid_index;
	};

}  // namespace caffe

#endif  // CAFFE_MTCNN_CLS_LAYER_HPP_
