from pathlib import Path
import hashlib
import json
from summarize_accuracy_evaluation import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/2026-10-20-real-photos'

def main():
    rows=[json.loads(s) for s in (OUT/'android-results.jsonl').read_text(encoding='utf-8').splitlines() if s]
    manifest=json.loads((OUT/'inputs/manifest.json').read_text(encoding='utf-8'))
    assert [r['id'] for r in rows] == [r['id'] for r in manifest]
    groups={str(side):[r for r in rows if r['max_side']==side] for side in [2400,1920]}
    result=dict(apk_sha256=hashlib.sha256((ROOT/'dist/ScoreCamera-1.2.0-debug.apk').read_bytes()).hexdigest(),
        distinct_photos=8, field_truth='Visually read four fields; user-confirmed totals in ascending file-number order',
        metrics={name:metrics(group) for name,group in groups.items()},
        cases=[{k:r.get(k) for k in ['id','expected','scores','total','error','elapsed_ms']} for r in rows])
    (OUT/'android-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
