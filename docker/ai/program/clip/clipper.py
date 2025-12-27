import math
import torch
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import sys

args = sys.argv

cutted_name = args[1]   #QR切り取り画像(入力)
train_data = args[2]    #学習重みデータ
clipped_name = args[3]  #矩形切り取り画像フォルダ(出力)

def clip(file_name, train_data, clipped_name):
    count = 0
    #重みデータ
    model = YOLO(train_data, verbose=False)
    #対象画像
    img = cv2.imread(file_name)
    #BGRをRGBに変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #閾値
    model.conf = 0.50
    results = model(file_name)

    for result in results:
        pos = result.boxes.xyxy
        num_list = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        for (point, num) in zip(pos, num_list):
            clipped = img[int(point[1]) : int(point[3]),int(point[0]) : int(point[2])]
            cv2.imwrite(clipped_name + "/" + str(count) + ".jpg", clipped)
            count = count + 1
        
    return 0

clip(cutted_name, train_data, clipped_name)