@Echo off
Echo caffe create_annoset Batch
:: https://github.com/conner99/caffe/blob/ssd-microsoft/tools/convert_annoset.cpp
:: You can modify the parameters in create_data.bat if needed.
:: It will create lmdb files for trainval and test with encoded original image:
::  - D:\caffe\data\VOC0712\trainval_lmdb
::  - D:\caffe\data\VOC0712\test_lmdb

set root_dir=D:\caffe
cd %root_dir%

set redo=1
set data_root_dir=data\VOC0712
set mapfile=%data_root_dir%\labelmap_voc.prototxt
set anno_type=detection
set db=lmdb
set min_dim=0
set max_dim=0
set width=0
set height=0

set "extra_cmd=--encode-type=jpg --encoded"

if %redo%==1 (
	set "extra_cmd=%extra_cmd% --redo"
)

for %%s in (trainval test) do (
echo Creating %%s lmdb...

python %root_dir%\scripts\create_annoset.py ^
	--anno-type=%anno_type% ^
	--label-map-file=%mapfile% ^
	--min-dim=%min_dim% ^
	--max-dim=%max_dim% ^
	--resize-width=%width% ^
	--resize-height=%height% ^
	--check-label %extra_cmd% ^
	%data_root_dir% ^
	%data_root_dir%\%%s.txt ^
	%data_root_dir%\%%s_%db%
)

pause