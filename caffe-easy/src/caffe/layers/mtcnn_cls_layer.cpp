#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/mtcnn_cls_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void MTCNNClsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		//label
		this->N = 0;
		this->valid_index.clear();
		const Dtype* label = bottom[1]->cpu_data();
		for (int i = 0; i < bottom[1]->count(); i++){
			if (label[i] != -1){
				this->N++;
				this->valid_index.push_back(i);
			}
		}

		top[0]->Reshape({ bottom[1]->count(), 2 });
		top[1]->Reshape({ bottom[1]->count(), 1 });
	}

	template <typename Dtype>
	void MTCNNClsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* top0 = top[0]->mutable_cpu_data();
		Dtype* top1 = top[1]->mutable_cpu_data();
		const Dtype* bot0 = bottom[0]->cpu_data();
		const Dtype* bot1 = bottom[1]->cpu_data();
		memset(top0, 0, top[0]->count() * sizeof(Dtype));
		memset(top1, 0, top[1]->count() * sizeof(Dtype));

		for (int i = 0; i < this->valid_index.size(); ++i){
			memcpy(top0+i*top[0]->channels(), bot0 + this->valid_index[i] * bottom[0]->channels(), bottom[0]->channels() * sizeof(Dtype));
			//memcpy(top1 + i, bot1 + this->valid_index[i] * bottom[1]->channels(), bottom[1]->channels() * sizeof(Dtype));
			//top0[i*top[0]->channels()+0] = bot0[this->valid_index[i]];
			top1[i] = bot1[this->valid_index[i]];
		}
	}

	template <typename Dtype>
	void MTCNNClsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		
		const Dtype* top0 = top[0]->cpu_diff();
		const Dtype* top1 = top[1]->cpu_diff();
		Dtype* bot0 = bottom[0]->mutable_cpu_diff();
		Dtype* bot1 = bottom[1]->mutable_cpu_diff();
		memset(bot0, 0, bottom[0]->count() * sizeof(Dtype));
		memset(bot1, 0, bottom[1]->count() * sizeof(Dtype));

		if (propagate_down[0] && this->N>0) {
			for (int i = 0; i < this->valid_index.size(); ++i){
				memcpy(bot0 + this->valid_index[i] * bottom[0]->channels(), top0 + i*top[0]->channels(), bottom[0]->channels() * sizeof(Dtype));
				//bot0[this->valid_index[i]] = top0[i];
				bot1[this->valid_index[i]] = top1[i];
				//memcpy(bot0 + this->valid_index[i] + i*bottom[0]->channels(), top0 + top[0]->channels()*i, bottom[0]->channels() * sizeof(Dtype));
				//memcpy(bot1 + this->valid_index[i] + i*bottom[1]->channels(), top1 + top[1]->channels()*i, bottom[0]->channels() * sizeof(Dtype));
				//memcpy(top1 + i, bot1 + this->valid_index[i] * bottom[1]->channels());
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(MTCNNClsLayer);
#endif

INSTANTIATE_CLASS(MTCNNClsLayer);
REGISTER_LAYER_CLASS(MTCNNCls);

}  // namespace caffe
