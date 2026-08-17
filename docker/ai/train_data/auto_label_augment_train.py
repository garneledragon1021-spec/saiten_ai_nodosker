import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


DIGIT_NAMES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="未アノテーション画像から疑似ラベルを作成し、増強データでYOLOを再学習する。"
    )
    parser.add_argument("--source-dir", default="box", help="未アノテーション画像が入ったフォルダー")
    parser.add_argument("--output-dir", default="docker/ai/train_data/generated", help="生成データセットの出力先")
    parser.add_argument("--runs-dir", default="docker/ai/train_data/runs", help="学習結果の出力先")
    parser.add_argument("--box-model", default="docker/ai/train_data/ref/best_cripper.pt", help="矩形検出の初期モデル")
    parser.add_argument("--digit-model", default="docker/ai/train_data/ref/best_detecter.pt", help="数字検出の初期モデル")
    parser.add_argument("--box-epochs", type=int, default=30, help="矩形検出モデルの学習エポック数")
    parser.add_argument("--digit-epochs", type=int, default=30, help="数字検出モデルの学習エポック数")
    parser.add_argument("--box-augment", type=int, default=5, help="矩形画像1枚あたりの増強枚数")
    parser.add_argument("--digit-augment", type=int, default=4, help="数字切り出し1枚あたりの増強枚数")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO学習時の入力サイズ")
    parser.add_argument("--batch", type=int, default=4, help="YOLO学習時のバッチサイズ")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--skip-train", action="store_true", help="データセット生成だけ実行する")
    return parser.parse_args()


def numeric_sort_key(path: Path):
    return int(path.stem) if path.stem.isdigit() else path.stem


def safe_reset_dir(path: Path):
    resolved = path.resolve()
    if "train_data/generated" not in str(resolved):
        raise ValueError(f"安全のため generated 配下以外は削除しません: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dataset_dirs(dataset_dir: Path):
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def detect_black_boxes(image):
    """黒い矩形枠をOpenCVで検出する。赤い数字は除外するため、RGBすべてが暗い画素だけを見る。"""
    height, width = image.shape[:2]
    b, g, r = cv2.split(image)
    mask = ((b < 115) & (g < 115) & (r < 115)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    image_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = w / max(h, 1)
        if area < image_area * 0.0012:
            continue
        if area > image_area * 0.04:
            continue
        if ratio < 0.35 or ratio > 4.0:
            continue
        boxes.append([x, y, x + w, y + h])

    return merge_boxes(boxes)


def merge_boxes(boxes, overlap_threshold=0.50):
    merged = []
    for box in sorted(boxes, key=lambda row: (row[1], row[0])):
        did_merge = False
        for index, current in enumerate(merged):
            if intersection_over_min_area(box, current) > overlap_threshold:
                merged[index] = [
                    min(box[0], current[0]),
                    min(box[1], current[1]),
                    max(box[2], current[2]),
                    max(box[3], current[3]),
                ]
                did_merge = True
                break
        if not did_merge:
            merged.append(box)
    return merged


def intersection_over_min_area(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / min(area_a, area_b)


def resize_with_boxes(image, boxes, max_dim):
    height, width = image.shape[:2]
    scale = min(max_dim / max(width, height), 1.0)
    if scale == 1.0:
        return image, boxes
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    scaled_boxes = [[coord * scale for coord in box] for box in boxes]
    return resized, scaled_boxes


def write_image_and_labels(image_path: Path, label_path: Path, image, boxes, class_ids):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    height, width = image.shape[:2]
    rows = []
    for class_id, box in zip(class_ids, boxes):
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        rows.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def clip_box(box, width, height, min_size=3):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, float(x1)))
    y1 = max(0, min(height - 1, float(y1)))
    x2 = max(0, min(width - 1, float(x2)))
    y2 = max(0, min(height - 1, float(y2)))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return [x1, y1, x2, y2]


def affine_augment(image, boxes, rng, *, allow_flip):
    height, width = image.shape[:2]
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.92, 1.08)
    tx = rng.uniform(-0.035, 0.035) * width
    ty = rng.uniform(-0.035, 0.035) * height
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += [tx, ty]

    augmented = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(230, 230, 230),
    )
    transformed_boxes = transform_boxes(boxes, matrix, width, height)

    if allow_flip and rng.random() < 0.25:
        augmented = cv2.flip(augmented, 1)
        transformed_boxes = [[width - b[2], b[1], width - b[0], b[3]] for b in transformed_boxes]

    alpha = rng.uniform(0.82, 1.22)
    beta = rng.uniform(-22, 22)
    augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

    if rng.random() < 0.35:
        noise = np.random.normal(0, rng.uniform(2, 8), augmented.shape).astype(np.int16)
        augmented = np.clip(augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.18:
        augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

    return augmented, transformed_boxes


def transform_boxes(boxes, matrix, width, height):
    transformed = []
    for x1, y1, x2, y2 in boxes:
        corners = np.array(
            [
                [x1, y1, 1],
                [x2, y1, 1],
                [x2, y2, 1],
                [x1, y2, 1],
            ],
            dtype=np.float32,
        )
        points = corners @ matrix.T
        new_box = [
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        ]
        clipped = clip_box(new_box, width, height)
        if clipped is not None:
            transformed.append(clipped)
    return transformed


def write_data_yaml(dataset_dir: Path, names):
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path


def build_box_dataset(source_paths, output_dir: Path, rng, augment_count):
    dataset_dir = output_dir / "box_detector"
    safe_reset_dir(dataset_dir)
    ensure_dataset_dirs(dataset_dir)

    stats = {
        "source_images": len(source_paths),
        "train_images": 0,
        "val_images": 0,
        "total_labels": 0,
        "empty_images": [],
        "label_counts": {},
    }

    for index, source_path in enumerate(source_paths):
        split = "val" if index % 5 == 0 else "train"
        image = cv2.imread(str(source_path))
        if image is None:
            continue
        boxes = detect_black_boxes(image)
        image, boxes = resize_with_boxes(image, boxes, max_dim=1600)
        class_ids = [0] * len(boxes)
        stem = source_path.stem

        write_image_and_labels(
            dataset_dir / "images" / split / f"{stem}.jpg",
            dataset_dir / "labels" / split / f"{stem}.txt",
            image,
            boxes,
            class_ids,
        )
        stats[f"{split}_images"] += 1
        stats["total_labels"] += len(boxes)
        stats["label_counts"][source_path.name] = len(boxes)
        if not boxes:
            stats["empty_images"].append(source_path.name)

        if split == "train" and boxes:
            for aug_index in range(augment_count):
                augmented, augmented_boxes = affine_augment(image, boxes, rng, allow_flip=True)
                write_image_and_labels(
                    dataset_dir / "images" / split / f"{stem}_aug{aug_index}.jpg",
                    dataset_dir / "labels" / split / f"{stem}_aug{aug_index}.txt",
                    augmented,
                    augmented_boxes,
                    [0] * len(augmented_boxes),
                )
                stats["train_images"] += 1
                stats["total_labels"] += len(augmented_boxes)

    stats["yaml"] = str(write_data_yaml(dataset_dir, {0: "box"}))
    return dataset_dir, stats


def crop_with_padding(image, box, padding_ratio=0.14):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    clipped = clip_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)
    if clipped is None:
        return None, None
    x1, y1, x2, y2 = map(int, clipped)
    return image[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


def detect_digit_boxes(model, crop):
    result = model(crop, conf=0.52, iou=0.45, imgsz=640, verbose=False)[0]
    boxes = []
    class_ids = []
    confidences = []
    for xyxy, cls, conf in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
    ):
        class_id = int(cls)
        confidence = float(conf)
        if confidence < 0.52:
            continue
        boxes.append([float(v) for v in xyxy])
        class_ids.append(class_id)
        confidences.append(confidence)

    keep = filter_digit_labels(boxes, class_ids, confidences, crop.shape[1], crop.shape[0])
    return [boxes[i] for i in keep], [class_ids[i] for i in keep], [confidences[i] for i in keep]


def filter_digit_labels(boxes, class_ids, confidences, width, height):
    candidates = []
    for index, box in enumerate(boxes):
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width / width < 0.035 or box_height / height < 0.12:
            continue
        if box_width / width > 0.65 or box_height / height > 0.95:
            continue
        candidates.append((index, confidences[index]))

    candidates.sort(key=lambda row: row[1], reverse=True)
    kept = []
    for index, _ in candidates:
        if len(kept) >= 3:
            break
        if all(box_iou(boxes[index], boxes[other]) < 0.45 for other in kept):
            kept.append(index)
    kept.sort(key=lambda index: boxes[index][0])
    return kept


def box_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / (area_a + area_b - intersection)


def build_digit_dataset(source_paths, output_dir: Path, rng, augment_count, digit_model_path: str):
    dataset_dir = output_dir / "digit_detector"
    safe_reset_dir(dataset_dir)
    ensure_dataset_dirs(dataset_dir)
    digit_model = YOLO(digit_model_path)

    stats = {
        "source_images": len(source_paths),
        "train_images": 0,
        "val_images": 0,
        "total_labels": 0,
        "skipped_crops": 0,
        "label_counts": {},
    }

    crop_index = 0
    for image_index, source_path in enumerate(source_paths):
        split = "val" if image_index % 5 == 0 else "train"
        image = cv2.imread(str(source_path))
        if image is None:
            continue
        box_labels = detect_black_boxes(image)
        for box_index, box in enumerate(box_labels):
            crop, _ = crop_with_padding(image, box)
            if crop is None:
                continue
            digit_boxes, class_ids, confidences = detect_digit_boxes(digit_model, crop)
            if not digit_boxes:
                stats["skipped_crops"] += 1
                continue

            name = f"{source_path.stem}_{box_index:02d}"
            write_image_and_labels(
                dataset_dir / "images" / split / f"{name}.jpg",
                dataset_dir / "labels" / split / f"{name}.txt",
                crop,
                digit_boxes,
                class_ids,
            )
            stats[f"{split}_images"] += 1
            stats["total_labels"] += len(digit_boxes)
            stats["label_counts"][name] = {
                "digits": len(digit_boxes),
                "classes": [DIGIT_NAMES[class_id] for class_id in class_ids],
                "confidences": [round(value, 4) for value in confidences],
            }
            crop_index += 1

            if split == "train":
                for aug_index in range(augment_count):
                    augmented, augmented_boxes = affine_augment(crop, digit_boxes, rng, allow_flip=False)
                    if len(augmented_boxes) != len(digit_boxes):
                        continue
                    write_image_and_labels(
                        dataset_dir / "images" / split / f"{name}_aug{aug_index}.jpg",
                        dataset_dir / "labels" / split / f"{name}_aug{aug_index}.txt",
                        augmented,
                        augmented_boxes,
                        class_ids,
                    )
                    stats["train_images"] += 1
                    stats["total_labels"] += len(augmented_boxes)

    stats["yaml"] = str(write_data_yaml(dataset_dir, {index: name for index, name in enumerate(DIGIT_NAMES)}))
    stats["crops_with_labels"] = crop_index
    return dataset_dir, stats


def train_model(model_path, data_yaml, runs_dir: Path, name: str, epochs: int, imgsz: int, batch: int):
    model = YOLO(model_path)
    result = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=0,
        project=str(runs_dir.resolve()),
        name=name,
        exist_ok=True,
        patience=max(3, min(10, epochs)),
        cache=False,
        plots=False,
    )
    save_dir = Path(result.save_dir)
    return {
        "save_dir": str(save_dir),
        "best": str(save_dir / "weights" / "best.pt"),
        "last": str(save_dir / "weights" / "last.pt"),
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    runs_dir = Path(args.runs_dir)
    source_paths = sorted(source_dir.glob("*.jpg"), key=numeric_sort_key)
    if not source_paths:
        raise FileNotFoundError(f"画像が見つかりません: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    box_dataset_dir, box_stats = build_box_dataset(source_paths, output_dir, rng, args.box_augment)
    digit_dataset_dir, digit_stats = build_digit_dataset(
        source_paths,
        output_dir,
        rng,
        args.digit_augment,
        args.digit_model,
    )

    report = {
        "box_dataset": box_stats,
        "digit_dataset": digit_stats,
        "training": {},
    }

    if not args.skip_train:
        report["training"]["box"] = train_model(
            args.box_model,
            box_dataset_dir / "data.yaml",
            runs_dir,
            "box_finetune",
            args.box_epochs,
            args.imgsz,
            args.batch,
        )
        report["training"]["digit"] = train_model(
            args.digit_model,
            digit_dataset_dir / "data.yaml",
            runs_dir,
            "digit_finetune",
            args.digit_epochs,
            args.imgsz,
            args.batch,
        )

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "report": str(report_path),
        "box_dataset": {
            "train_images": box_stats["train_images"],
            "val_images": box_stats["val_images"],
            "total_labels": box_stats["total_labels"],
            "empty_images": box_stats["empty_images"],
            "yaml": box_stats["yaml"],
        },
        "digit_dataset": {
            "train_images": digit_stats["train_images"],
            "val_images": digit_stats["val_images"],
            "total_labels": digit_stats["total_labels"],
            "skipped_crops": digit_stats["skipped_crops"],
            "crops_with_labels": digit_stats["crops_with_labels"],
            "yaml": digit_stats["yaml"],
        },
        "training": report["training"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
