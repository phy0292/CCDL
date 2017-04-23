import os
import random

trainacc = 0.9


fs = os.listdir("xwk_st/JPEGImages")
ntrain = int(trainacc * len(fs))
nval = len(fs) - ntrain
random.shuffle(fs)


with open("trainval.txt", "wb") as tf:
    for i in range(ntrain):
        p = fs[i].rfind(".")
        tf.write(fs[i][:p] + "\n")


pos = ntrain
with open("test.txt", "wb") as tf:
    for i in range(nval):
        p = fs[i+pos].rfind(".")
        tf.write(fs[i+pos][:p] + "\n")




