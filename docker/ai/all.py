import os
import sys
import argparse
import json
import glob
import cv2
import qrcode
import traceback

# pyzbar はネイティブの zbar ライブラリを必要とする。
# macOS では pyenv の Python が Homebrew のライブラリディレクトリを
# 自動で検索しないため、zbar の検索パスを追加する。
if sys.platform == "darwin":
    # Apple Silicon と Intel Mac の Homebrew 標準インストール先を対象にする。
    homebrew_lib_dirs = ("/opt/homebrew/lib", "/usr/local/lib")
    # 利用者が指定した既存の検索パスは保持する。
    fallback_lib_dirs = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(os.pathsep)
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = os.pathsep.join(
        directory for directory in (*homebrew_lib_dirs, *fallback_lib_dirs) if directory
    )

from pyzbar.pyzbar import decode
from ultralytics import YOLO

def read_qr_image(input_path, cutted_path):
    """画像内の指定QRコードを確認し、QRコードより下の採点対象部分を切り出す。"""
    img = cv2.imread(input_path)
    if img is None:
        return "Different"

    decoded_qr = decode(img)
    if not decoded_qr:
        return "Different"

    h, w = img.shape[:2]

    for obj in decoded_qr:
        data = obj.data.decode("utf-8")
        if data != "fujishima startup QRcode":
            continue

        xmin = max(obj.rect.left, 0)
        ymin = max(obj.rect.top + obj.rect.height, 0)
        xmax = w
        ymax = h

        cut_img = img[ymin:ymax, xmin:xmax]
        if cut_img.size == 0:
            return "Different"

        os.makedirs(os.path.dirname(cutted_path), exist_ok=True)
        cv2.imwrite(cutted_path, cut_img)
        return data

    return "Different"


def clip_images(cutted_path, train_data_path, clipped_folder):
    """答案の切り出し画像から設問ごとの領域を検出し、左から順に保存する。"""
    os.makedirs(clipped_folder, exist_ok=True)
    model = YOLO(train_data_path, verbose=False)

    img = cv2.imread(cutted_path)
    if img is None:
        raise FileNotFoundError(f"cutted image not found: {cutted_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model.conf = 0.50
    results = model(cutted_path)

    data = []
    for result in results:
        pos = result.boxes.xyxy
        for point in pos:
            clipped = img_rgb[int(point[1]):int(point[3]), int(point[0]):int(point[2])]
            data.append([clipped, float(point[0])])

    data.sort(key=lambda x: x[1])

    for i, (clipped_img, _) in enumerate(data):
        save_path = os.path.join(clipped_folder, f"{i}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(clipped_img, cv2.COLOR_RGB2BGR))


def numCheck(data_str):
    """YOLOが返した英語の数字ラベルを整数へ変換する。未対応のラベルはNoneを返す。"""
    mapping = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    return mapping.get(data_str)


def sort_numbers(org, count, pos):
    """検出した数字を画像内の左端座標順に並べ替える。"""
    combined = list(zip(org[:count], pos[:count]))
    combined.sort(key=lambda x: x[1])
    return [row[0] for row in combined]


def res_number(data, count):
    """左から並んだ最大3桁の数字リストを、ひとつの得点値に変換する。"""
    if count == 0:
        return 0
    if count == 1:
        return data[0]
    if count == 2:
        return data[0] * 10 + data[1]
    return data[0] * 100 + data[1] * 10 + data[2]


def detect(file_name, train_data):
    """1つの設問画像から数字を検出し、読み取った得点を整数で返す。"""
    print(f"[DEBUG detect] Starting detection for: {file_name}")
    model = YOLO(train_data, verbose=False)

    img = cv2.imread(file_name)
    if img is None:
        print(f"[DEBUG detect] image not found: {file_name}")
        return 0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model.conf = 0.75
    results = model(file_name)

    count = 0
    data = [0, 0, 0]
    pos = [0, 0, 0]

    for result in results:
        point_list = result.boxes.xyxy
        num_list = [result.names[cls.item()] for cls in result.boxes.cls.int()]

        for (point, num) in zip(point_list, num_list):
            val = numCheck(num)
            if val is None:
                continue
            data[count] = val
            pos[count] = float(point[0])
            count += 1
            if count >= 3:
                break
        if count >= 3:
            break

    score_random = sort_numbers(data, count, pos)
    result_score = res_number(score_random, count)
    print(f"[DEBUG detect] Final score: {result_score}")
    return result_score


def detect_call(clipped_folder, train_data):
    """設問ごとに保存された画像を順番に採点し、得点一覧を返す。"""
    file_list = glob.glob(os.path.join(clipped_folder, "*.jpg"))
    file_list.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    results = []
    for img_path in file_list:
        score = detect(img_path, train_data)
        results.append(score)

    return results


def save_result(result_path, scores):
    """設問番号をキーとした得点一覧をJSONファイルへ保存する。"""
    new_data = {f"question{i+1}": str(score) for i, score in enumerate(scores)}
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)


def prefer_finetuned_model(finetuned_path, fallback_path):
    """再学習済みモデルが存在する場合はそれを使い、無い場合は従来モデルを使う。"""
    if os.path.exists(finetuned_path):
        return finetuned_path
    return fallback_path


def main():
    """入力画像のQR確認、設問切り出し、採点、結果JSONの保存を順に実行する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name')
    parser.add_argument('--base-dir', default=os.path.dirname(__file__))
    args = parser.parse_args()

    file_name = args.file_name
    base_dir = args.base_dir

    # 再学習済みモデルがあれば優先して利用する。
    train_data_box = prefer_finetuned_model(
        os.path.join(base_dir, "train_data/ref/best_cripper_finetuned.pt"),
        os.path.join(base_dir, "train_data/ref/best_cripper.pt")
    )
    train_data_score = prefer_finetuned_model(
        os.path.join(base_dir, "train_data/ref/best_detecter_finetuned.pt"),
        os.path.join(base_dir, "train_data/ref/best_detecter.pt")
    )

    org_file = os.path.normpath(os.path.join(base_dir, "../images/origin", f"{file_name}.jpg"))
    cutted_file = os.path.normpath(os.path.join(base_dir, "../images/cutted", f"{file_name}.jpg"))
    clipped_folder = os.path.normpath(os.path.join(base_dir, "../images/clipped", file_name))
    result_path = os.path.normpath(os.path.join(base_dir, "../result", f"{file_name}.json"))

    if not os.path.exists(org_file):
        print("error: input image not found:", org_file)
        sys.exit(1)

    os.makedirs(os.path.dirname(cutted_file), exist_ok=True)
    os.makedirs(clipped_folder, exist_ok=True)

    # 元画像のQRコードを確認し、採点対象の答案部分を切り出す。
    print(f"[DEBUG] Running read_qr_image with: {org_file}")
    qr_text = read_qr_image(org_file, cutted_file)
    print(f"[DEBUG] read_qr_image result: {qr_text}")

    if qr_text == "fujishima startup QRcode":
        # 切り出した答案から、設問ごとの解答欄を画像として分割する。
        print(f"[DEBUG] Running clip_images with: {cutted_file}")
        clip_images(cutted_file, train_data_box, clipped_folder)
        print("[DEBUG] clip_images completed")

        # 各解答欄の数字を読み取り、設問ごとの得点を取得する。
        print(f"[DEBUG] Running detect on: {clipped_folder}")
        scores = detect_call(clipped_folder, train_data_score)
        print("[DEBUG] detect completed")

        # 取得した得点をJSON形式の結果ファイルへ保存する。
        save_result(result_path, scores)
        print(f"[DEBUG] Result saved to: {result_path}")
    else:
        print("skip!!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)
