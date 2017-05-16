#pragma once

struct layer_node {
	const caffe::LayerParameter *layer;
	const layer_node *next;  // top-side
	const layer_node *prev;  // bottom-side

	layer_node() : layer(0), next(0), prev(0) {}
	explicit layer_node(const caffe::LayerParameter *l)
		: layer(l), next(0), prev(0) {}
};

class caffe_layer_vector {
public:
	explicit caffe_layer_vector(const caffe::NetParameter &net_orig)
		: net(net_orig) {
		if (net.layers_size() > 0) {
			upgradev1net(net_orig, &net);
		}

		nodes.reserve(net.layer_size());

		for (int i = 0; i < net.layer_size(); i++) {
			auto &l = net.layer(i);

			if (layer_table.find(l.name()) != layer_table.end()) continue;

			nodes.emplace_back(&l);
			layer_table[l.name()] = &nodes.back();
		}

		for (size_t i = 0; i < nodes.size(); i++) {
			auto &l = nodes[i];

			if (l.layer->bottom_size() > 0 && blob_table[l.layer->bottom(0)]) {
				auto &bottom = blob_table[l.layer->bottom(0)];
				l.prev = bottom;
				layer_table[bottom->layer->name()]->next = &l;
			}

			if (l.layer->top_size() > 0) {
				blob_table[l.layer->top(0)] = &l;
			}
		}

		auto root = std::find_if(nodes.begin(), nodes.end(),
			[](const layer_node &n) { return n.prev == 0; });

		if (root == nodes.end()) {
			throw "root layer not found";
		}

		root_node = &*root;
		const layer_node *current = &*root;

		while (current) {
			node_list.push_back(current->layer);
			current = current->next;
		}
	}

	size_t size() const { return node_list.size(); }

	const caffe::LayerParameter &operator[](size_t index) const {
		return *(node_list[index]);
	}

public:
	static const char *v1type2name(caffe::V1LayerParameter_LayerType type){
		switch (type) {
		case caffe::V1LayerParameter_LayerType_NONE: return "";
		case caffe::V1LayerParameter_LayerType_ABSVAL: return "AbsVal";
		case caffe::V1LayerParameter_LayerType_ACCURACY: return "Accuracy";
		case caffe::V1LayerParameter_LayerType_ARGMAX: return "ArgMax";
		case caffe::V1LayerParameter_LayerType_BNLL: return "BNLL";
		case caffe::V1LayerParameter_LayerType_CONCAT: return "Concat";
		case caffe::V1LayerParameter_LayerType_CONTRASTIVE_LOSS:
			return "ContrastiveLoss";
		case caffe::V1LayerParameter_LayerType_CONVOLUTION: return "Convolution";
		case caffe::V1LayerParameter_LayerType_DECONVOLUTION:
			return "Deconvolution";
		case caffe::V1LayerParameter_LayerType_DATA: return "Data";
		case caffe::V1LayerParameter_LayerType_DROPOUT: return "Dropout";
		case caffe::V1LayerParameter_LayerType_DUMMY_DATA: return "DummyData";
		case caffe::V1LayerParameter_LayerType_EUCLIDEAN_LOSS:
			return "EuclideanLoss";
		case caffe::V1LayerParameter_LayerType_ELTWISE: return "Eltwise";
		case caffe::V1LayerParameter_LayerType_EXP: return "Exp";
		case caffe::V1LayerParameter_LayerType_FLATTEN: return "Flatten";
		case caffe::V1LayerParameter_LayerType_HDF5_DATA: return "HDF5Data";
		case caffe::V1LayerParameter_LayerType_HDF5_OUTPUT: return "HDF5Output";
		case caffe::V1LayerParameter_LayerType_HINGE_LOSS: return "HingeLoss";
		case caffe::V1LayerParameter_LayerType_IM2COL: return "Im2col";
		case caffe::V1LayerParameter_LayerType_IMAGE_DATA: return "ImageData";
		case caffe::V1LayerParameter_LayerType_INFOGAIN_LOSS:
			return "InfogainLoss";
		case caffe::V1LayerParameter_LayerType_INNER_PRODUCT:
			return "InnerProduct";
		case caffe::V1LayerParameter_LayerType_LRN: return "LRN";
		case caffe::V1LayerParameter_LayerType_MEMORY_DATA: return "MemoryData";
		case caffe::V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS:
			return "MultinomialLogisticLoss";
		case caffe::V1LayerParameter_LayerType_MVN: return "MVN";
		case caffe::V1LayerParameter_LayerType_POOLING: return "Pooling";
		case caffe::V1LayerParameter_LayerType_POWER: return "Power";
		case caffe::V1LayerParameter_LayerType_RELU: return "ReLU";
		case caffe::V1LayerParameter_LayerType_SIGMOID: return "Sigmoid";
		case caffe::V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS:
			return "SigmoidCrossEntropyLoss";
		case caffe::V1LayerParameter_LayerType_SILENCE: return "Silence";
		case caffe::V1LayerParameter_LayerType_SOFTMAX: return "Softmax";
		case caffe::V1LayerParameter_LayerType_SOFTMAX_LOSS:
			return "SoftmaxWithLoss";
		case caffe::V1LayerParameter_LayerType_SPLIT: return "Split";
		case caffe::V1LayerParameter_LayerType_SLICE: return "Slice";
		case caffe::V1LayerParameter_LayerType_TANH: return "TanH";
		case caffe::V1LayerParameter_LayerType_WINDOW_DATA: return "WindowData";
		case caffe::V1LayerParameter_LayerType_THRESHOLD: return "Threshold";
		default: throw "unknown v1 layer-type";
		}
	}

	static void upgradev1layer(const caffe::V1LayerParameter &old,
		caffe::LayerParameter *dst) {
		dst->Clear();

		for (int i = 0; i < old.bottom_size(); i++) {
			dst->add_bottom(old.bottom(i));
		}

		for (int i = 0; i < old.top_size(); i++) {
			dst->add_top(old.top(i));
		}

		if (old.has_name()) dst->set_name(old.name());
		if (old.has_type()) dst->set_type(v1type2name(old.type()));

		for (int i = 0; i < old.blobs_size(); i++) {
			dst->add_blobs()->CopyFrom(old.blobs(i));
		}

		for (int i = 0; i < old.param_size(); i++) {
			while (dst->param_size() <= i) dst->add_param();
			dst->mutable_param(i)->set_name(old.param(i));
		}

#define COPY_PARAM(name)        \
		if (old.has_##name##_param()) \
		dst->mutable_##name##_param()->CopyFrom(old.name##_param())

		COPY_PARAM(accuracy);
		COPY_PARAM(argmax);
		COPY_PARAM(concat);
		COPY_PARAM(contrastive_loss);
		COPY_PARAM(convolution);
		COPY_PARAM(data);
		COPY_PARAM(dropout);
		COPY_PARAM(dummy_data);
		COPY_PARAM(eltwise);
		COPY_PARAM(exp);
		COPY_PARAM(hdf5_data);
		COPY_PARAM(hdf5_output);
		COPY_PARAM(hinge_loss);
		COPY_PARAM(image_data);
		COPY_PARAM(infogain_loss);
		COPY_PARAM(inner_product);
		COPY_PARAM(lrn);
		COPY_PARAM(memory_data);
		COPY_PARAM(mvn);
		COPY_PARAM(pooling);
		COPY_PARAM(power);
		COPY_PARAM(relu);
		COPY_PARAM(sigmoid);
		COPY_PARAM(softmax);
		COPY_PARAM(slice);
		COPY_PARAM(tanh);
		COPY_PARAM(threshold);
		COPY_PARAM(window_data);
		COPY_PARAM(transform);
		COPY_PARAM(loss);
#undef COPY_PARAM
	}

	static void upgradev1net(const caffe::NetParameter &old,
		caffe::NetParameter *dst){
		dst->CopyFrom(old);
		dst->clear_layers();
		dst->clear_layer();

		for (int i = 0; i < old.layers_size(); i++) {
			upgradev1layer(old.layers(i), dst->add_layer());
		}
	}

	caffe::NetParameter net;
	layer_node *root_node;
	/* layer name -> layer */
	std::map<std::string, layer_node *> layer_table;
	/* blob name -> bottom holder */
	std::map<std::string, layer_node *> blob_table;
	std::vector<layer_node> nodes;
	std::vector<const caffe::LayerParameter *> node_list;
};