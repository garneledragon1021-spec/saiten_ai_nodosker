"""Separate coverage from silent errors; do not treat rejection as a correct reading."""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/2026-10-20'

def metrics(rows):
    fields = correct = wrong = unread = complete = exact = wrong_complete = total_wrong = errors = 0
    timings = []
    for row in rows:
        expected = row['expected']
        actual = row.get('scores', [None] * len(expected))
        fields += len(expected)
        errors += int('error' in row)
        for target, value in zip(expected, actual):
            if value is None: unread += 1
            elif value == target: correct += 1
            else: wrong += 1
        if all(value is not None for value in actual):
            complete += 1
            if actual == expected: exact += 1
            else: wrong_complete += 1
            if sum(actual) != sum(expected): total_wrong += 1
        timings.append(row['elapsed_ms'])
    accepted = correct + wrong
    return dict(cases=len(rows), fields=fields, correct_fields=correct, wrong_fields=wrong, unread_fields=unread,
        accepted_field_accuracy=round(correct/accepted, 6) if accepted else None,
        correct_field_coverage=round(correct/fields, 6) if fields else None,
        complete_sheets=complete, exact_sheets=exact, wrong_complete_sheets=wrong_complete,
        wrong_totals=total_wrong, failed_pipeline_cases=errors,
        median_ms=statistics.median(timings), max_ms=max(timings))

def main():
    rows = [json.loads(line) for line in (OUT/'results.jsonl').read_text(encoding='utf-8').splitlines() if line]
    manifest = json.loads((OUT/'inputs/manifest.json').read_text(encoding='utf-8'))
    assert [r['id'] for r in rows] == [r['id'] for r in manifest], 'Incomplete or mismatched evaluation run'
    groups = defaultdict(list)
    for row in rows: groups[row['family']].append(row)
    truth = list(csv.DictReader((OUT/'manual_truth.csv').open(encoding='utf-8')))
    label_differences = []
    for t in truth:
        path = ROOT/'docker/ai/train_data/generated/digit_detector/labels/val'/f"{t['id']}.txt"
        digits = sorted([line.split() for line in path.read_text().splitlines() if line], key=lambda s:float(s[1]))
        label = int(''.join(s[0] for s in digits)) if digits else None
        if label != int(t['score']): label_differences.append(dict(id=t['id'], manual=int(t['score']), pseudo=label))
    summary = dict(apk_sha256=hashlib.sha256((ROOT/'dist/ScoreCamera-1.2.0-debug.apk').read_bytes()).hexdigest(),
        truth_sha256=hashlib.sha256((OUT/'manual_truth.csv').read_bytes()).hexdigest(),
        overall=metrics(rows),
        constructed_only=metrics([r for r in rows if r['family'].startswith('constructed-')]),
        automatic_qr_only=metrics([r for r in rows if r['family'] != 'existing-screenshot-injected']),
        by_family={name:metrics(group) for name,group in sorted(groups.items())},
        pseudo_label_differences=label_differences,
        pipeline_errors=dict(Counter(r.get('error') for r in rows if 'error' in r)),
        wrong_cases=[dict(id=r['id'],expected=r['expected'],scores=r.get('scores')) for r in rows
            if any(a is not None and a != b for a,b in zip(r.get('scores',[]),r['expected']))])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','family','expected','scores','error','elapsed_ms'])
        for r in rows: writer.writerow([r['id'],r['family'],r['expected'],r.get('scores'),r.get('error',''),r['elapsed_ms']])
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
