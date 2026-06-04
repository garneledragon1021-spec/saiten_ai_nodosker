import cv2
from ultralytics import YOLO
import sys
import json
import glob
import os
import traceback
#import math

print("[DEBUG detect_new.py] Script started", flush=True)

args = sys.argv
print(f"[DEBUG detect_new.py] args: {args}", flush=True)

train_data = args[1]    #学習重みデータ
result_name = args[2]   #点数結果出力ファイル(json)
clipped_folder = args[3]  #矩形切り取り画像フォルダ

print(f"[DEBUG detect_new.py] train_data: {train_data}", flush=True)
print(f"[DEBUG detect_new.py] result_name: {result_name}", flush=True)
print(f"[DEBUG detect_new.py] clipped_folder: {clipped_folder}", flush=True)

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
    
    print(f"[DEBUG detect] Starting detection for: {file_name}")
    print(f"[DEBUG detect] Model path: {train_data}")
    
    count=0
    data = [0, 0, 0]
    pos = [0,0,0]
    
    #重みデータ
    print("[DEBUG detect] Loading YOLO model...")
    model = YOLO(train_data, verbose=False)
    print("[DEBUG detect] YOLO model loaded")
    
    #対象画像
    print("[DEBUG detect] Reading image...")
    img = cv2.imread(file_name)
    print(f"[DEBUG detect] Image shape: {img.shape if img is not None else 'None'}")
    
    #BGRをRGBに変換.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    #閾値
    print("[DEBUG detect] Running inference...")
    model.conf = 0.75
    results = model(file_name)
    print(f"[DEBUG detect] Inference complete, results: {len(results)}")
    
    #数字の個数分繰り返す
    for result in results:
        
        #数字の座標
        point_list = result.boxes.xyxy
        
        #それぞれ画像のクラスIDを取得し,numCheckへ渡すためリストへ
        num_list = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        
        print(f"[DEBUG detect] Found {len(num_list)} numbers: {num_list}")
        
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
    print(f"[DEBUG detect] Final score: {result}")
    return result

def detect_call ():
    
    #ファイルはこれやで
    print(f"[DEBUG] clipped_folder = {clipped_folder}")
    file_list = glob.glob(os.path.join(clipped_folder, "*.jpg"))
    print(f"[DEBUG] Found {len(file_list)} images")
    
    # ファイル名でソート
    file_list.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    
    result = []
    point_sum = 0
    
    #矩形の回数分繰り返す
    for i, img_path in enumerate(file_list):
        print(f"[DEBUG] Processing image {i+1}/{len(file_list)}: {img_path}")
        #数字検出
        score = detect(img_path, train_data)
        print(f"[DEBUG] Detected score: {score}")
        #大問ごとの点数をリストへ格納
        result.append(score)
        point_sum += score

    result.append(point_sum)

    #json出力
    print("[DEBUG] Writing JSON result")
    json_Write(result)
    print("[DEBUG] Done")
    
    return 0

#json出力関数
def json_Write(result):
    #リストの内容をディクショナリに変換
    new_data = {f"question{i+1}": str(score) for i, score in enumerate(result)}
    
    #jsonをopenして書き込み
    with open(result_name, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)
        
    return 0

try:
    print("[DEBUG detect_new.py] Calling detect_call()", flush=True)
    detect_call()
    print("[DEBUG detect_new.py] detect_call() completed successfully", flush=True)
except Exception as e:
    print(f"[ERROR detect_new.py] Exception occurred: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)