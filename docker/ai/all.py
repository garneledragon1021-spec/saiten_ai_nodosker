import os
import sys
import argparse
import json
import glob
import cv2
import qrcode
import traceback
from pyzbar.pyzbar import decode
from ultralytics import YOLO


def read_qr_image(input_path, cutted_path):
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
    mapping = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    return mapping.get(data_str)


def sort_numbers(org, count, pos):
    combined = list(zip(org[:count], pos[:count]))
    combined.sort(key=lambda x: x[1])
    return [row[0] for row in combined]


def res_number(data, count):
    if count == 0:
        return 0
    if count == 1:
        return data[0]
    if count == 2:
        return data[0] * 10 + data[1]
    return data[0] * 100 + data[1] * 10 + data[2]


def detect(file_name, train_data):
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
    file_list = glob.glob(os.path.join(clipped_folder, "*.jpg"))
    file_list.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    results = []
    for img_path in file_list:
        score = detect(img_path, train_data)
        results.append(score)

    return results


def save_result(result_path, scores):
    new_data = {f"question{i+1}": str(score) for i, score in enumerate(scores)}
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name')
    parser.add_argument('--base-dir', default=os.path.dirname(__file__))
    args = parser.parse_args()

    file_name = args.file_name
    base_dir = args.base_dir

    train_data_box = os.path.join(base_dir, "program/clip/ref/best.pt")
    train_data_score = os.path.join(base_dir, "program/detect/ref/best.pt")

    org_file = os.path.normpath(os.path.join(base_dir, "../images/origin", f"{file_name}.jpg"))
    cutted_file = os.path.normpath(os.path.join(base_dir, "../images/cutted", f"{file_name}.jpg"))
    clipped_folder = os.path.normpath(os.path.join(base_dir, "../images/clipped", file_name))
    result_path = os.path.normpath(os.path.join(base_dir, "../result", f"{file_name}.json"))

    if not os.path.exists(org_file):
        print("error: input image not found:", org_file)
        sys.exit(1)

    os.makedirs(os.path.dirname(cutted_file), exist_ok=True)
    os.makedirs(clipped_folder, exist_ok=True)

    print(f"[DEBUG] Running read_qr_image with: {org_file}")
    qr_text = read_qr_image(org_file, cutted_file)
    print(f"[DEBUG] read_qr_image result: {qr_text}")

    if qr_text == "fujishima startup QRcode":
        print(f"[DEBUG] Running clip_images with: {cutted_file}")
        clip_images(cutted_file, train_data_box, clipped_folder)
        print("[DEBUG] clip_images completed")

        print(f"[DEBUG] Running detect on: {clipped_folder}")
        scores = detect_call(clipped_folder, train_data_score)
        print("[DEBUG] detect completed")

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
