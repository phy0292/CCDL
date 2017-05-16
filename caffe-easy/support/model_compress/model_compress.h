


#pragma once

#include <support-common.h>


//infile：输入的模型文件路径
//fileOutPath：保存的模型文件路径
//upLevel：压缩的指标，这个实则是在对权重做变换，例如：w' = ((int)(w * upLevel)) / upLevel;
//         通常取值10000，比如nin取值到300，建议自己实际测试下精度
Caffe_API bool __stdcall model_compress(const char* infile, const char* fileOutPath, float upLevel);