@Echo off
Echo caffe get_image_size Batch
:: https://github.com/conner99/caffe/blob/ssd-microsoft/tools/get_image_size.cpp
:: This program retrieves the sizes of a set of images.
:: Usage:
::   get_image_size [FLAGS] ROOTFOLDER/ LISTFILE OUTFILE
::
:: where ROOTFOLDER is the root folder that holds all the images and
:: annotations, and LISTFILE should be a list of files as well as their labels
:: or label files.
:: For classification task, the file should be in the format as
::   imgfolder1/img1.JPEG 7
::   ....
:: For detection task, the file should be in the format as
::   imgfolder1/img1.JPEG annofolder1/anno1.xml
::   ....

set TOOLS=D:\caffe\Build\x64\Release
set DATA=D:\caffe\data\VOC0712

%TOOLS%\get_image_size ^
	%Data%\ ^
	%Data%\test.txt ^
	%Data%\test_name_size.txt
	
pause