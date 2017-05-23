@echo off
echo Creating MNIST lmdb...

rd /q /s train_lmdb
rd /q /s val_lmdb
mkdir train_lmdb
mkdir val_lmdb
"../packageRelease/x64-gpu-cuda8.0/convert_mnist_data.exe" mnist_data/train-images-idx3-ubyte mnist_data/train-labels-idx1-ubyte train_lmdb --backend=lmdb
"../packageRelease/x64-gpu-cuda8.0/convert_mnist_data.exe" mnist_data/t10k-images-idx3-ubyte mnist_data/t10k-labels-idx1-ubyte val_lmdb --backend=lmdb

echo Done.
pause