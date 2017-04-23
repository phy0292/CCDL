@Echo off
Echo caffe ssd_pascal Batch
:: https://github.com/conner99/caffe

:: cd caffe_root
cd ../../

:: Skript-Ordner
python %~dp0\ssd_pascal.py

pause