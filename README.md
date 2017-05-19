# CCDL
深度学习应用框架，专注于深度学习的工程应用方面<br/>
为了更容易的使用和部署深度学习，极简主义，easy to use. easy!!!<br/>
目前集中精力在windows下，将来也会支持linux系统<br/>

## 编译工具
visual studio 2013，建议使用这个版本。目前编译全部是静态编译方式，最后也仅仅依赖几个OpenBlas的dll(该DLL在CPU上可以实现大约3-5倍的加速，所以很有必要)或者甚至不依赖dll，即可发布您的软件<br/>

## 特性：
添加了[Center Loss](https://github.com/ydwen/caffe-face)，以支持人脸识别的训练<br/>
添加了[MTCNN](https://github.com/happynear/MTCNN_face_detection_alignment)，以支持人脸检测的实现<br/>
添加了[SSD](https://github.com/weiliu89/caffe/tree/ssd)，以支持对目标检测的支持<br/>
添加了多标签支持，以实现多任务或OCR、验证码类的识别任务<br/>
添加了Win32、x64，CPU、GPU的支持，方便应用到各个领域上<br/>
添加了调用的cpp接口、c语言接口，实现分类和其他识别任务，以实现支持例如：C#、C++、C、易语言等快速开发<br/>
添加了[模型压缩](https://github.com/dlunion/CCDL/tree/master/caffe-easy/support/model_compress)的方法，实现对训练模型的压缩工作，目前nin网络实现5倍的压缩，精度损失极小。<br/>
添加了[多图模式](https://github.com/dlunion/CCDL/blob/master/caffe-easy/support/classification/classification.cpp)，使得轻易就能够搭建高性能识别服务器<br/>
添加了[任务池](https://github.com/dlunion/CCDL/blob/master/caffe-easy/support/classification/task_pool.cpp)，满载GPU很轻松<br/>