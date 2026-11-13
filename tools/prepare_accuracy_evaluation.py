"""Prepare a reproducible visual audit; existing pseudo-labels are NOT ground truth."""
from pathlib import Path
from PIL import Image, ImageDraw
import csv
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation' / '2026-10-20'
SOURCE = ROOT / 'docker/ai/train_data/generated/digit_detector/images/val'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(SOURCE.glob('*.jpg'), key=lambda p: tuple(map(int, p.stem.split('_'))))
    for page in range((len(paths) + 39) // 40):
        sheet = Image.new('RGB', (1200, 8 * 176), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(paths[page * 40:(page + 1) * 40]):
            x, y = (index % 5) * 240, (index // 5) * 176
            crop = Image.open(path).convert('RGB')
            crop.thumbnail((220, 145))
            sheet.paste(crop, (x + (240 - crop.width) // 2, y + 24))
            draw.text((x + 8, y + 4), path.stem, fill='black')
        sheet.save(OUT / f'audit-{page + 1}.png')
    print(f'{len(paths)} existing validation crops; contact sheets: {OUT}')

    truth_path = OUT / 'manual_truth.csv'
    if not truth_path.exists():
        return
    truth = list(csv.DictReader(truth_path.open(encoding='utf-8')))
    assert {r['id'] for r in truth} == {p.stem for p in paths}
    sample = cv2.imread(str(ROOT / 'app/src/androidTest/assets/sample-answer.jpg'))
    _, points, _ = cv2.QRCodeDetector().detectAndDecode(sample)
    assert points is not None
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    qr = cv2.resize(sample[y-12:y+h+12, x-12:x+w+12], (120, 120), interpolation=cv2.INTER_NEAREST)
    inputs = OUT / 'inputs'
    inputs.mkdir(exist_ok=True)
    manifest = []

    def save_case(name, family, image, expected, ids, injected=None):
        cv2.imwrite(str(inputs / f'{name}.png'), image)
        row = dict(id=name, family=family, file=f'{name}.png', expected=expected, source_ids=ids)
        if injected is not None:
            row['qr'] = injected
        manifest.append(row)

    # Four fields per constructed row, last five together (no unsupported singleton).
    groups = [truth[i:i+4] for i in range(0, 88, 4)] + [truth[88:]]
    for index, group in enumerate(groups):
        sheet = np.full((410, 190 + 210 * len(group), 3), 255, np.uint8)
        sheet[20:140, 20:140] = qr
        for j, row in enumerate(group):
            crop = Image.open(SOURCE / (row['id'] + '.jpg')).convert('RGB')
            crop = crop.rotate(int(row['rotate_ccw']), expand=True)
            crop.thumbnail((180, 180), Image.Resampling.LANCZOS)
            patch = np.asarray(crop)[:, :, ::-1]
            y0, x0 = 180 + (180-patch.shape[0])//2, 180+j*210+(180-patch.shape[1])//2
            sheet[y0:y0+patch.shape[0], x0:x0+patch.shape[1]] = patch
        expected = [int(row['score']) for row in group]
        ids = [row['id'] for row in group]
        gradient = np.linspace(.30, 1., sheet.shape[1])[None, :, None]
        _, encoded = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 35])
        variants = dict(baseline=sheet,
            low_resolution=cv2.resize(sheet, (480, int(sheet.shape[0]*480/sheet.shape[1])), interpolation=cv2.INTER_AREA),
            dark=(sheet.astype(np.float32)*.45).astype(np.uint8),
            shadow=(sheet.astype(np.float32)*gradient).astype(np.uint8),
            blur=cv2.GaussianBlur(sheet, (0,0), 1.5),
            jpeg35=cv2.imdecode(encoded, cv2.IMREAD_COLOR))
        for condition, image in variants.items():
            save_case(f'row{index+1:02d}-{condition}', 'constructed-'+condition, image, expected, ids)

    # Existing whole-sheet fixture: camera/gallery-sized inputs; no injected QR.
    for longest in [2400, 1920, 1280, 1080, 720]:
        scale = longest/max(sample.shape[:2])
        image = cv2.resize(sample, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        save_case(f'sample-{longest}', 'existing-whole-sheet', image, [12,95,84], ['sample-answer'])
    screenshot = cv2.imread(str(ROOT / 'app/src/androidTest/assets/reported-live.png'))
    save_case('reported-auto-qr', 'existing-screenshot-auto', screenshot, [6,13,42,10], ['reported-live'])
    save_case('reported-injected-qr', 'existing-screenshot-injected', screenshot, [6,13,42,10], ['reported-live'],
        [[17,1137],[77,1137],[77,1200],[17,1200]])
    (inputs / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(manifest)} cases; {len(truth)} manually audited crop values; inputs: {inputs}')

if __name__ == '__main__':
    main()
