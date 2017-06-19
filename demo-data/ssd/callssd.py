#coding=gbk
import sys
sys.path.insert(0, r"E:/project/3.github/CCDL2/CCDL/caffe-easy/python")
import caffe
import time
import numpy as np
import cv2

modelFile = "deploy.prototxt"
pretrained = "../SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel"
reload(sys)
sys.setdefaultencoding("utf-8")

caffe.set_mode_gpu();

net = caffe.Classifier(modelFile, pretrained,channel_swap=(2, 1, 0), raw_scale=256)

f = open("list_file.txt" ,"r").read().split("\n")
f = [(f[i].split(" ")[0], f[i]) for i in range(len(f))]

for i in range(len(f)):
    file = f[i][0]
    input_image = caffe.io.load_image(file)
    r = net.predict([input_image], False)
    cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR, input_image)

    boxs = r[0][0]
    #Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
    for k in range(len(boxs)):
        imwidth = input_image.shape[1]
        imheight = input_image.shape[0]
        box = boxs[k]
        label = box[1]
        score = box[2]
        xmin = int(box[3] * imwidth)
        ymin = int(box[4] * imheight)
        xmax = int(box[5] * imwidth)
        ymax = int(box[6] * imheight)

        if score > 0.25:
            cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
    cv2.imshow("abc", input_image)
    key = cv2.waitKey()
    if(key & 0xFF == 0x1B):
        exit()