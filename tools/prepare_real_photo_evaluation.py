"""Copy only the eight explicitly supplied photos; preserve original JPEG bytes."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evaluation/2026-10-20-real-photos'

def main():
    truth = json.loads((OUT/'truth.json').read_text(encoding='utf-8'))
    inputs = OUT/'inputs'
    inputs.mkdir(exist_ok=True)
    manifest = []
    hashes = {}
    for row in truth['photos']:
        n = row['number']
        assert sum(row['scores']) == row['total']
        source = Path.home()/'Downloads'/f'Image ({n}).jpe'
        target = inputs/f'user-photo-{n:02d}.jpg'
        shutil.copyfile(source, target)
        hashes[str(n)] = hashlib.sha256(target.read_bytes()).hexdigest()
        for side in [2400, 1920]:
            manifest.append(dict(id=f'user{n:02d}-{side}', family=f'real-photo-{side}',file=target.name,
                expected=row['scores'], user_total=row['total'], source_ids=[f'Image ({n}).jpe'],max_side=side))
    (inputs/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(f'Copied 8 supplied photos; {len(manifest)} cases. Android decodes/resizes using the app gallery API.')

if __name__ == '__main__': main()
