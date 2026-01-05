import math
# import torch
import cv2
# from matplotlib import pyplot as plt
from ultralytics import YOLO
import sys
import json
#import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

args = sys.argv

clipped_name = args[1]  #矩形切り取り画像
train_data = args[2]    #学習重みデータ
result_name = args[3]   #点数結果出力ファイル(json)
number = "question" + str(args[4])    #大問番号

#文字➡数字変換
def numCheck(data):
    if data=="zero":
        return 0
    elif data=="one":
        return 1
    elif data=="two":
        return 2
    elif data=="three":
        return 3
    elif data=="four":
        return 4
    elif data=="five":
        return 5
    elif data=="six":
        return 6
    elif data=="seven":
        return 7
    elif data=="eight":
        return 8
    elif data=="nine":
        return 9

#桁毎に正しく並べ直す
def sort(org, count, pos):
    i=0
    j=0

    data = [0,0,0]
    detect = 0
    max = 0
    count = count - 1 

    while j < 3:
        for i in range(len(pos)):
            if pos[i] > max:
                detect = i
                max=pos[i]
        
        data[count-j]=org[detect]
        pos[detect]=0
        j=j+1
        max=0

    return data

#値を示す
def res(data, count):
    ans=0
    #もし点数が一桁なら・・・
    if count==1:
        ans = data[0]
    elif count==2:
        ans = data[0] * 10 + data[1]
    elif count==3:
        ans = data[0] * 100 + data[1] * 10 + data[2]
    
    return ans

def detect(file_name, train_data):
    
    count=0
    data = [0, 0, 0]
    pos = [0,0,0]
    #重みデータ
    model = YOLO(train_data, verbose=False)
    #対象画像
    img = cv2.imread(file_name)
    #BGRをRGBに変換.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    #閾値
    model.conf = 0.75
    results = model(file_name)
    
    for result in results:
        
        #矩形の座標
        point_list = result.boxes.xyxy
        
        #それぞれ画像のクラスIDを取得し,numCheckへ渡すためリストへ
        num_list = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        
        #zip関数でpoint_listとnum_listをペアに（タプルに）
        for (point, num) in zip(point_list, num_list):
            
            data[count] = numCheck(num)  
            xmin = point[0]

            if data[count]==None:
                count = count - 1
            else:
                pos[count] = xmin

            count=count+1

    data = sort(data, count, pos)
    result = res(data, count)
    
    #結果出力
    return result

res = detect(clipped_name, train_data)

new_data = {number: { 'score': str(res)}}

if number=="question1":
    with open(result_name, "w") as f:
        json.dump(new_data, f, indent=2)
else:
    with open(result_name, "r") as f:
        read_data = json.load(f)

    save_data = [read_data, new_data]
    with open(result_name, "w") as f:
        json.dump(save_data, f, indent=2)