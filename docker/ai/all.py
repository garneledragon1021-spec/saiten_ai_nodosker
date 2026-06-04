import subprocess   #他プログラム呼び出し
import os           #ファイル操作
import sys          #外部引数取得


# 安全に子プロセスを実行し、失敗時は詳細を出して終了する

def run_checked(command):
    env = os.environ.copy()
    # pyzbar が macOS(Homebrew) の zbar を見つけられるよう探索パスを補う
    fallback = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    homebrew_lib = "/opt/homebrew/lib"
    env["DYLD_FALLBACK_LIBRARY_PATH"] = homebrew_lib if not fallback else f"{fallback}:{homebrew_lib}"

    res = subprocess.run(command, capture_output=True, text=True, env=env)
    
    # サブプロセスの出力を表示
    if res.stdout:
        print(res.stdout, end='', flush=True)
    if res.stderr:
        print(res.stderr, end='', flush=True)
    
    if res.returncode != 0:
        print("error:", " ".join(command))
        sys.exit(res.returncode)
    return res



#外部入力
args = sys.argv
if len(args) < 2:
    print("usage: python3 docker/ai/all.py <file_name_without_ext>")
    sys.exit(1)

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
exec_detect = os.path.join(base_dir, "program/detect/detect_new.py")        #detect_new.pyパス

if not os.path.exists(org_file):
    print("error: input image not found:", org_file)
    sys.exit(1)

#切り取り画像保存フォルダ作成
os.makedirs(clipped_folder, exist_ok=True)

#QR.py実行（python3 QR.py 元画像パス 切り取り画像パス）
print(f"[DEBUG] Running QR.py with: {org_file}")
qr_res = run_checked([exec_7seg, exec_qr, org_file, cutted_file])
qr_text = (qr_res.stdout or "").strip()
print(f"[DEBUG] QR.py result: {qr_text}")

#QR.pyの結果が正しければ
if qr_text == "fujishima startup QRcode":
    #clipper.py実行（python3 clipper.py 切り取り画像パス 学習データパス 矩形切り取り画像保存フォルダ）
    print(f"[DEBUG] Running clipper.py with: {cutted_file}")
    run_checked([exec_7seg, exec_box, cutted_file, train_data_box, clipped_folder])
    print("[DEBUG] clipper.py completed")

    #detect.py実行（python3 detect_new.py 学習データパス 点数jsonファイルパス 矩形切り取り画像保存フォルダ）
    print(f"[DEBUG] Running detect_new.py with: {clipped_folder}")
    run_checked([exec_7seg, exec_detect, train_data_score, result_path, clipped_folder])
    print("[DEBUG] detect_new.py completed")

#例外
else:
    print("skip!!")