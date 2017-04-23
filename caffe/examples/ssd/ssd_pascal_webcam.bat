@Echo off
Echo caffe ssd_pascal_webcam Batch
:: https://github.com/conner99/caffe

:: cd caffe_root
cd D:\caffe

:: Skript-Ordner
python %~dp0\ssd_pascal_webcam.py

pause