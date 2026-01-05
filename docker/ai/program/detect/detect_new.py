import math
import cv2
from ultralytics import YOLO
import sys
import json
import glob
import os

args = sys.argv

train_data = args[1]    #学習重みデータ
result_name = args[2]   #点数結果出力ファイル(json)
clipped_folder = args[3]  #矩形切り取り画像フォルダ

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
    #使っている桁だけをペアにする（初期値の0を混ぜない）
    combined = list(zip(org[:count], pos[:count]))
    
    #座標をもとにソートしてしまえ！（小さい順＝左から右へ）
    combined.sort(key=lambda x: x[1])
    
    #ソート後、数字データ（一行目）だけ取り出してしまえ！
    data = [row[0] for row in combined]

    return data

#値を示す
def res(data, count):
    ans=0
    if count==1:
        ans = data[0]
    elif count==2:
        ans = data[0] * 10 + data[1]
    elif count==3:
        ans = data[0] * 100 + data[1] * 10 + data[2]
    
    return ans

#数字検出
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
    
    #数字の個数分繰り返す
    for result in results:
        
        #数字の座標
        point_list = result.boxes.xyxy
        
        #それぞれ画像のクラスIDを取得し,numCheckへ渡すためリストへ
        num_list = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        
        #zipで数字と座標をペアに（タプルに）
        for (point, num) in zip(point_list, num_list):
            
            #画像のIDを数字へ変換,dataへ格納
            data[count] = numCheck(num)  
            
            #リスト0番目の数字の座標をxminへ格納
            xmin = point[0]

            #もし数字が認識されなければもう一度
            if data[count]==None:
                continue
            else:
                #数字ごとの座標を格納
                pos[count] = xmin

            #インクリメントさ。
            count += 1

    #並び替え(数字,桁数,座標)
    score_random = sort(data, count, pos)
    #整数データへ(並び替えられた数字,桁数)
    result = res(score_random, count)
    
    #結果出力
    return result

def detect_call ():
    
    #ファイルはこれやで
    file_list = glob.glob(os.path.join(clipped_folder, "*.jpg"))
    # ファイル名(拡張子除く)を数値として昇順ソート（0.jpg,1.jpg,2.jpg...）
    file_list.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    
    result = []
    
    for img_path in file_list:
        score = detect(img_path, train_data)
        result.append(score)
        
    json_Write(result)
    
    return 0


def json_Write(result):
    new_data = {f"question{i+1}": str(score) for i, score in enumerate(result)}
    with open(result_name, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)
        
detect_call()