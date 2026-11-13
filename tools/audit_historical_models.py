"""Read historical checkpoint provenance and export a separate comparison model, never replace app assets."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/2026-10-20-real-photos'

def main():
    torch.set_num_threads(2)
    report = {}
    for name in ['best_detecter.pt','best_detecter_finetuned.pt','best_cripper.pt','best_cripper_finetuned.pt']:
        path = ROOT/'docker/ai/train_data/ref'/name
        model = YOLO(str(path))
        ckpt = model.ckpt
        args = ckpt.get('train_args', {})
        report[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            git_blob=subprocess.check_output(['git','hash-object',str(path)],cwd=ROOT,text=True).strip(),
            pre_ios_blob=subprocess.check_output(['git','rev-parse',f'c99d471:docker/ai/train_data/ref/{name}'],cwd=ROOT,text=True).strip(),
            date=ckpt.get('date'), version=ckpt.get('version'), epoch=ckpt.get('epoch'),
            train_args={k:args.get(k) for k in ['model','data','epochs','imgsz','name','project']},
            classes=model.names)
        print(name, json.dumps(report[name],ensure_ascii=False),flush=True)
    (OUT/'historical-model-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    cached = ROOT/'.model-cache/manual-base-score.pt'
    if not cached.exists(): shutil.copyfile(ROOT/'docker/ai/train_data/ref/best_detecter.pt',cached)
    if not cached.with_suffix('.onnx').exists():
        YOLO(str(cached)).export(format='onnx',imgsz=640,opset=17,dynamic=False,simplify=False,nms=False,device='cpu')
    print('Historical comparison ONNX:',cached.with_suffix('.onnx'),flush=True)

if __name__ == '__main__': main()
