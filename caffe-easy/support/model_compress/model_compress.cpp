#include <support-common.h>
#include <io.h>
#include <fcntl.h>
#include <cv.h>
#include <highgui.h>
#include <fstream>  // NOLINT(readability/streams)
#include "caffe.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include <iosfwd>
#include <stdio.h>
#include <iostream>
#include "caffe_layer_vector.h"
#include <import-staticlib.h>

using namespace caffe;
using namespace std;
using namespace cv;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::GzipOutputStream;
using google::protobuf::Message;

extern "C"{
	void compressNet(NetParameter& net, float up = 1000){
		float intup = up;
		float floatup = up == 0 ? 1 : (1 / (float)up);
		net.mutable_state()->set_phase(caffe::Phase::TEST);

		if (net.layers_size() > 0){
			NetParameter tmp;
			tmp.CopyFrom(net);
			caffe_layer_vector::upgradev1net(tmp, &net);
		}

		for (int i = 1; i < net.layer_size(); ++i){
			LayerParameter& param = *net.mutable_layer(i);
			//printf("layer: %s\n", param.name().c_str());

#if 0
			if (param.mutable_blobs()->size()){
				BlobProto* blob = param.mutable_blobs(0);
				BlobProto* bais = param.mutable_blobs(1);
				float* data = blob->mutable_data()->mutable_data();
				int len = blob->data_size();
				//printf("处理层：data[%d][%d][%s]  [%d]\n", param.mutable_blobs()->size(), len, param.name().c_str(), bais->data_size());

				float* weights = data;
				float* bias = bais->mutable_data()->mutable_data();
				int bias_len = bais->data_size();

				for (int k = 0; k < bias_len; ++k)
					bias[k] = ((int)(bias[k] * intup)) * floatup;

				for (int k = 0; k < len; ++k)
					weights[k] = ((int)(weights[k] * intup)) * floatup;
			}
#endif

			for (int j = 0; j < param.mutable_blobs()->size(); ++j){
				BlobProto* blob = param.mutable_blobs(j);
				float* data = blob->mutable_data()->mutable_data();
				int len = blob->data_size();
				float* weights = data;
				for (int k = 0; k < len; ++k)
					weights[k] = ((int)(weights[k] * intup)) * floatup;

#if 0
				BlobProto* bais = param.mutable_blobs(1);
				float* bias = bais->mutable_data()->mutable_data();
				int bias_len = bais->data_size();

				for (int k = 0; k < bias_len; ++k)
					bias[k] = ((int)(bias[k] * intup)) * floatup;
#endif
			}
		}
	}

	//, int saveToNormal
	Caffe_API bool __stdcall model_compress(const char* infile, const char* fileOutPath, float upLevel){
		int fd = _open(infile, O_RDONLY | O_BINARY);
		if (fd == -1) return false;

		std::shared_ptr<ZeroCopyInputStream> raw_input = std::make_shared<FileInputStream>(fd);
		std::shared_ptr<CodedInputStream> coded_input = std::make_shared<CodedInputStream>(raw_input.get());
		coded_input->SetTotalBytesLimit(INT_MAX, 536870912);

		NetParameter net;
		bool success = net.ParseFromCodedStream(coded_input.get());
		coded_input.reset();
		raw_input.reset();
		_close(fd);
		if (!success) return false;
		
		compressNet(net, upLevel);

		//if (saveToNormal)
		{
			fstream output(fileOutPath, ios::out | ios::trunc | ios::binary);
			success = net.SerializePartialToOstream(&output);
		}
#if 0
		else{
			int fd2 = _open(fileOutPath, O_RDWR | O_BINARY | O_TRUNC | O_CREAT, S_IREAD | S_IWRITE);
			if (fd2 == -1) return false;

			ZeroCopyOutputStream* raw_outpt = new FileOutputStream(fd2);
			//GzipOutputStream::Options op;
			//op.compression_level = 9;
			//op.format = GzipOutputStream::Format::ZLIB;

			GzipOutputStream* coded_output = new GzipOutputStream(raw_outpt);
			success = net.SerializeToZeroCopyStream(coded_output);
			delete coded_output;
			delete raw_outpt;
			_close(fd2);
		}
#endif
		return success;
	}
}

int main(int argc, char** argv){

	//这个压缩程序产生的ys.caffemodel文件，可以使用压缩算法把模型压缩的很小，因为原始的模型你无论怎么压缩
	//尺寸很难减小，而这个算法处理后的模型，可以很轻易减小很大
	//但是这个指标upLevel，需要注意，取值越大，压缩效果越差，但是精度损失越少
	//相反，取值越小，压缩效果越好，精度损失越大
	//他只是对权重做一个变换：w = ((int)(w * upLevel)) / upLevel
	const char* caffemodel = "../../../../../demo-data/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel";
	const char* caffemodelsave = "../../../../../demo-data/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.ys.caffemodel";
	if (!model_compress(caffemodel, caffemodelsave, 1000)){
		printf("压缩失败.\n");
	}
	return 0;
}