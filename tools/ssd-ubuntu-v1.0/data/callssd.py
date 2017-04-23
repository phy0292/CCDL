#coding=gbk
import sys
sys.path.insert(0, r"/hope/userdata/dl/5.ssd/caffe/python")
import caffe
import time
import numpy as np
import cv2

modelFile = "deploy.prototxt"
pretrained = "VGG_VOC0712_SSD_300x300_iter_50420.caffemodel"
reload(sys)
sys.setdefaultencoding("utf-8")

caffe.set_mode_gpu();

net = caffe.Classifier(modelFile, pretrained,channel_swap=(2, 1, 0), raw_scale=256)

f = open("list_file" ,"r").read().split("\n")
del f[-1]
f = [(f[i].split(" ")[0], f[i]) for i in range(len(f))]

for i in range(len(f)):
    file = f[i][0]
    input_image = caffe.io.load_image(file)
    r = net.predict([input_image], False)

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

        if score > 0.85:
            cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR, input_image)
            cv2.rectangle(input_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.imshow("abc", input_image)
            key = cv2.waitKey()
            if(key & 0xFF == 0x1B):
                exit()