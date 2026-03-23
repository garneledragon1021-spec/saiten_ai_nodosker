import cv2                          #画像処理関係
from ultralytics import YOLO        #YOLOモデル関係
import sys                          #外部引数取得 
#from matplotlib import pyplot as plt
#import torch
#import math

args = sys.argv

cutted_name = args[1]   #QR切り取り画像(入力)
train_data = args[2]    #学習重みデータ
clipped_name = args[3]  #矩形切り取り画像フォルダ(出力)


def clip(file_name, train_data, clipped_name):

    #重みデータ
    model = YOLO(train_data, verbose=False)
    #対象画像
    img = cv2.imread(file_name)
    #BGRをRGBに変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #閾値
    model.conf = 0.50
    results = model(file_name)
    
    data = []

    #矩形の数繰り返す
    for result in results:
        
        #座標取得・格納
        pos = result.boxes.xyxy
    
        #切り取られた画像と座標をリストへ格納
        for point in pos: 
            clipped = img[int(point[1]):int(point[3]), int(point[0]):int(point[2])]
            data.append([clipped, point[0]])
            
    #x座標でソート
    data.sort(key=lambda x: x[1])
    
    #切り取り画像を連番で保存
    for i, (clipped_img, x_pos) in enumerate(data, start=0):
        save_path = f"{clipped_name}/{i}.jpg"
        cv2.imwrite(save_path, clipped_img)
        
    return 0

clip(cutted_name, train_data, clipped_name)