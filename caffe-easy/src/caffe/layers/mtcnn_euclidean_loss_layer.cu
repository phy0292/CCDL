#include <vector>

#include "caffe/layers/mtcnn_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MTCNNEuclideanLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::Reshape(bottom, top);
	CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
		<< "Inputs must have the same dimension.";
	diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MTCNNEuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  //label
  const Dtype* label = bottom[2]->cpu_data();
  Dtype* diff = diff_.mutable_gpu_data();
  int countLabel = bottom[2]->num();
  int channel = bottom[0]->channels();
  caffe_gpu_memset(sizeof(Dtype)*count, 0, diff);

  const Dtype* b0 = bottom[0]->gpu_data();
  const Dtype* b1 = bottom[1]->gpu_data();
  Dtype loss = 0;

  for (int i = 0; i < countLabel; ++i){
	  if (label[i] != 0){
		  caffe_gpu_sub(
			  channel,
			  b0 + i * channel,
			  b1 + i * channel,
			  diff + i * channel);
		  Dtype dot;
		  caffe_gpu_dot(channel, diff + i * channel, diff + i * channel, &dot);
		  loss += dot / channel / Dtype(2);
	  }
  }

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MTCNNEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* label = bottom[2]->cpu_data();
	int countLabel = bottom[2]->num();
	int channels = bottom[0]->channels();
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			caffe_gpu_memset(sizeof(Dtype)*bottom[i]->count(), 0, bottom[i]->mutable_gpu_diff());

			const Dtype sign = (i == 0) ? 1 : -1;
			for (int j = 0; j < countLabel; ++j){
				const Dtype alpha = sign * top[0]->cpu_diff()[0];
				if (label[j] != 0){
					caffe_gpu_axpby(
						channels,							// count
						alpha,                              // alpha
						diff_.gpu_data() + channels * j,                   // a
						Dtype(0),                           // beta
						bottom[i]->mutable_gpu_diff() + channels * j);  // b
				}
			}
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(MTCNNEuclideanLossLayer);

}  // namespace caffe
