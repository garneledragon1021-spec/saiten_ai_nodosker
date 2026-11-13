"""Isolated digit-model comparison on frozen manually located fields, not end-to-end app accuracy."""
from pathlib import Path
import json
import cv2
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/2026-10-20-real-photos'

def detect(session,image):
    h,w=image.shape[:2]; scale=min(640/w,640/h)
    sw,sh=max(1,int(w*scale)),max(1,int(h*scale));px,py=(640-sw)/2,(640-sh)/2
    boxed=cv2.warpAffine(image,np.float32([[sw/w,0,px],[0,sh/h,py]]),(640,640),flags=cv2.INTER_LINEAR,borderValue=(114,114,114))
    tensor=boxed[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255
    raw=session.run(None,{session.get_inputs()[0].name:tensor})[0][0].T
    classes=raw[:,4:].argmax(1); confidence=raw[np.arange(len(raw)),classes+4]
    bounds=np.column_stack(((raw[:,0]-raw[:,2]/2-px)/scale,(raw[:,1]-raw[:,3]/2-py)/scale,
        (raw[:,0]+raw[:,2]/2-px)/scale,(raw[:,1]+raw[:,3]/2-py)/scale))
    bounds[:,[0,2]]=bounds[:,[0,2]].clip(0,w);bounds[:,[1,3]]=bounds[:,[1,3]].clip(0,h)
    pending=list(np.where(confidence>=.45)[0][np.argsort(-confidence[confidence>=.45])]);kept=[]
    while pending:
        best=pending.pop(0);kept.append(best);b=bounds[best];rest=[]
        for i in pending:
            a=bounds[i];inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
            if inter/max(1,union)<=.45:rest.append(i)
        pending=rest
    kept=[i for i in kept if .035<=(bounds[i,2]-bounds[i,0])/w<=.7 and .2<=(bounds[i,3]-bounds[i,1])/h<=.95 and .08*w<=(bounds[i,0]+bounds[i,2])/2<=.92*w]
    kept=sorted(sorted(kept,key=lambda i:-confidence[i])[:3],key=lambda i:bounds[i,0])
    return [dict(digit=int(classes[i]),confidence=float(confidence[i])) for i in kept]

def main():
    options=ort.SessionOptions();options.intra_op_num_threads=2
    models={name:ort.InferenceSession(str(path),options) for name,path in {
        'original':ROOT/'.model-cache/manual-base-score.onnx',
        'auto_finetuned':ROOT/'app/src/main/assets/score_detector.onnx'}.items()}
    truth=json.loads((OUT/'truth.json').read_text(encoding='utf-8'))['photos']
    rois=json.loads((OUT/'field_rois.json').read_text())['photos']
    crops=OUT/'manual-field-crops';crops.mkdir(exist_ok=True)
    results=[]
    for photo in truth:
        n=photo['number'];image=cv2.imread(str(OUT/f'inputs/user-photo-{n:02d}.jpg'))
        ratio=2400/image.shape[0];image=cv2.resize(image,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
        factor=2400/1824
        for field,(coords,expected) in enumerate(zip(rois[str(n)],photo['scores']),1):
            left,top,right,bottom=[int(v*factor) for v in coords]
            dx=int((right-left)*.1);dy=int((bottom-top)*.18)
            patch=image[max(0,top-dy):min(image.shape[0],bottom+dy),max(0,left-dx):min(image.shape[1],right+dx)]
            cv2.imwrite(str(crops/f'{n:02d}-{field}.png'),patch)
            adjusted=np.clip(patch.astype(np.float32)*1.35+128*(1-1.35)+6,0,255).astype(np.uint8)
            row=dict(photo=n,field=field,expected=expected)
            for name,session in models.items():
                variants=[detect(session,p) for p in [patch,adjusted]]
                digits=max(variants,key=lambda ds:len(ds)*2+sum(d['confidence'] for d in ds))
                row[name]=dict(value=int(''.join(str(d['digit']) for d in digits)) if digits else None,digits=digits)
            results.append(row)
            print(json.dumps(row),flush=True)
    summary={}
    for name in models:
        correct=sum(r[name]['value']==r['expected'] for r in results)
        unread=sum(r[name]['value'] is None for r in results)
        exact=sum(all(r[name]['value']==r['expected'] for r in results if r['photo']==p['number']) for p in truth)
        summary[name]=dict(fields=len(results),correct=correct,wrong=len(results)-correct-unread,unread=unread,exact_photos=exact)
    (OUT/'digit-model-comparison.json').write_text(json.dumps(dict(scope='Manually located fields, approximate production YOLO preprocessing, no QR/frame/CNN/OCR/score-resolver evaluation',summary=summary,results=results),ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__': main()
