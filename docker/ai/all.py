import subprocess   #他プログラム呼び出し
import os           #ファイル操作
import glob         #ファイル取得
import sys          #外部引数取得
import re           #文字列操作

#外部入力
args = sys.argv

file_name = args[1]     #画像（入力）

#誰でも動かせるように相対参照しているよ
base_dir = os.path.dirname(__file__)
train_data_box = os.path.join(base_dir, "program/clip/ref/best.pt")                         #学習データ(矩形検出)
train_data_score = os.path.join(base_dir, "program/detect/ref/best.pt")                     #学習データ(数字検出)

org_file = os.path.join(base_dir, "../images/origin/" + file_name + ".jpg")          #画像パス
cutted_file = os.path.join(base_dir, "../images/cutted/" + file_name + ".jpg")       #QR.py処理後画像パス
clipped_folder = os.path.join(base_dir, "../images/clipped/" + file_name)            #clipper.py処理後画像ファイルパス
result_path = os.path.join(base_dir, "../result/" + file_name + ".json")             #点数jsonファイルパス（出力）

exec_7seg = "python3"          #実行時コマンド用

exec_qr = os.path.join(base_dir, "program/qr/QR.py")                        #QR.pyパス
exec_box = os.path.join(base_dir, "program/clip/clipper.py")                #clipper.pyパス

#切り取り画像保存フォルダ作成
if not os.path.isdir(clipped_folder):
    os.mkdir(clipped_folder)

exec_detect = os.path.join(base_dir, "program/detect/detect_new.py")        #detect_new.pyパス

#QR.py実行（python3 QR.py 元画像パス 切り取り画像パス）
throwcommand = exec_7seg + " " + exec_qr + " " + org_file + " " + cutted_file

#出力
qr_res = subprocess.run(throwcommand, shell=True, capture_output=True)

#print(str(qr_res.stdout))

#QR.pyの結果が正しければ
if re.match("fujishima startup QRcode", str(qr_res.stdout)[2:-2]):
    #clipper.py実行（python3 clipper.py 切り取り画像パス 学習データパス 矩形切り取り画像保存フォルダ）
    throwcommand = exec_7seg + " " + exec_box + " " + cutted_file + " " + train_data_box + " " + clipped_folder
    subprocess.run(throwcommand, shell=True, capture_output=False)
    
    #detect.py実行（python3 detect_new.py 矩形切り取り画像保存フォルダ 学習データパス 点数jsonファイルパス）
    throwcommand = exec_7seg + " " + exec_detect + " " + train_data_score + " " + result_path + " " + clipped_folder
    subprocess.run(throwcommand, shell=True, text=True)

#例外        
elif str(qr_res.stdout) == "Different":
    print("skip!!")