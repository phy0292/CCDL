
import os


with open("labelmap_coco.prototxt", "rb") as f:
    arr = f.read().split("\n")

    out = open("labels.txt", "wb")
    for i in range(3, len(arr), 5):
        b = arr[i].find('"')
        e = arr[i].find('"', b+1)
        name = arr[i][b+1:e]
        out.write(name + "\n")
    out.close()