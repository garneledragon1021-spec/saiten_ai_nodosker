#Pythonファイル実行用ファイル
import subprocess
import os
import glob
import sys
import re


#初期設定
args = sys.argv

file_name = args[1]     #指定ファイル名

org_file = "/home/sou/saiten_ai_nodocker/docker/images/origin/" + file_name + ".jpg"    #画像ファイル名(仮)
cutted_file = "/home/sou/saiten_ai_nodocker/docker/images/cutted/" + file_name + ".jpg"  #QRコード検出・切り取り処理
clipped_folder = "/home/sou/saiten_ai_nodocker/docker/images/clipped/" + file_name   #矩形切り取り処理画像フォルダ
result_path = "/home/sou/saiten_ai_nodocker/docker/result/" + file_name + ".json"   #点数データ格納用ファイルパス
base_dir = os.path.dirname(__file__)
train_data_box = os.path.join(base_dir, "program/clip/ref/best.pt")   #学習データ(矩形検出)
train_data_score = os.path.join(base_dir, "program/detect/ref/best.pt")   #学習データ(数字検出)

exec_7seg = "python3"

#QRコード用プログラム
exec_qr = os.path.join(base_dir, "program/qr/QR.py")

#矩形枠用プログラム
exec_box = os.path.join(base_dir, "program/clip/clipper.py")
if not os.path.isdir(clipped_folder):
    os.mkdir(clipped_folder)

#数字検出プログラム
exec_detect = os.path.join(base_dir, "program/detect/detect.py")

#QRコード用プログラム実行
#throwcommand = os.path.abspath(exec_7seg) + " " + exec_qr + " " + org_file + " " + cutted_file
throwcommand = exec_7seg + " " + exec_qr + " " + org_file + " " + cutted_file
qr_res = subprocess.run(throwcommand, shell=True, capture_output=True)

# print("STDOUT:", repr(qr_res.stdout))
# print("STDERR:", repr(qr_res.stderr))
# print("RETURN CODE:", qr_res.returncode)

print("通過")

print(str(qr_res.stdout))

if re.match("fujishima startup QRcode", str(qr_res.stdout)[2:-2]):
    #矩形枠用プログラム実行
    throwcommand = exec_7seg + " " + exec_box + " " + cutted_file + " " + train_data_box + " " + clipped_folder
    subprocess.run(throwcommand, shell=True)
    
    #数字検出プログラム実行
    index = 1
    file_list = glob.glob(os.path.join(clipped_folder, "*.jpg"))
    
    #大問の個数分繰り返す
    for img_path in file_list:
        #detect.py呼び出し
        throwcommand = exec_7seg + " " + exec_detect + " " + img_path + " " + train_data_score + " " + result_path + " " + str(index)
        result = subprocess.run(throwcommand, shell=True, text=True)

        #現在の大問番号
        index = index + 1
        #print(result.stdout)
        #出力データは、テキストファイルのカンマ区切り or JSONファイル
elif str(qr_res.stdout) == "Different":
    print("skip!!")