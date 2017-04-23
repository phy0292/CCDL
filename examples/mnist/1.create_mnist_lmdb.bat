@echo off
echo Creating MNIST lmdb...

rd /q /s train_lmdb
rd /q /s val_lmdb

"../../caffe_cpu/caffe/convert_mnist_data.exe" mnist_data/train-images-idx3-ubyte mnist_data/train-labels-idx1-ubyte train_lmdb --backend=lmdb
"../../caffe_cpu/caffe/convert_mnist_data.exe" mnist_data/t10k-images-idx3-ubyte mnist_data/t10k-labels-idx1-ubyte val_lmdb --backend=lmdb

echo Done.
pause