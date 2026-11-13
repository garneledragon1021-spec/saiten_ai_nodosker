"""Compare frozen 1.2.0 measurements with final 1.3.0 runs, preserving old evidence."""
from pathlib import Path
import hashlib
import json
from summarize_accuracy_evaluation import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/2026-10-20-v1.3'

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

def compare(old,new):
    assert [r['id'] for r in old] == [r['id'] for r in new], 'Case ordering or membership changed'
    assert [r['expected'] for r in old] == [r['expected'] for r in new], 'Ground truth changed'
    return dict(before=metrics(old),after=metrics(new))

def main():
    old_photos=read(ROOT/'evaluation/2026-10-20-real-photos/android-results.jsonl')
    photos=read(OUT/'real-photos.jsonl')
    old_broad=read(ROOT/'evaluation/2026-10-20/results.jsonl')
    broad=read(OUT/'broad-cases.jsonl')
    result=dict(apk_sha256=hashlib.sha256((ROOT/'app/build/outputs/apk/debug/app-debug.apk').read_bytes()).hexdigest(),
        limitation='Known development/regression images, not an independent holdout or Motorola camera benchmark',
        real_photos={str(side):compare([r for r in old_photos if r['max_side']==side],[r for r in photos if r['max_side']==side]) for side in [2400,1920]},
        broad_all=compare(old_broad,broad),
        constructed=compare([r for r in old_broad if r['family'].startswith('constructed-')],[r for r in broad if r['family'].startswith('constructed-')]),
        broad_automatic_qr=compare([r for r in old_broad if r['family']!='existing-screenshot-injected'],[r for r in broad if r['family']!='existing-screenshot-injected']),
        broad_by_family={family:compare([r for r in old_broad if r['family']==family],[r for r in broad if r['family']==family]) for family in sorted({r['family'] for r in broad})},
        remaining_wrong=[r for r in photos+broad if any(a is not None and a!=b for a,b in zip(r.get('scores',[]),r['expected']))],
        real_photo_cases=[{k:r.get(k) for k in ['id','expected','scores','total','error']} for r in photos])
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['broad_by_family','real_photo_cases']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
