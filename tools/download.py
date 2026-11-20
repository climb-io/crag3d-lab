#!/usr/bin/env python3
"""照 data/manifest.json 從 Hugging Face 資料集下載檔案，檢查 SHA-256；沒有任何相依套件。

用法：
  python3 tools/download.py                          全部（約 4.6 GB）
  python3 tools/download.py --only models/ours legacy-2d   只下載這些資料夾開頭的路徑
  python3 tools/download.py --out data/hf            放到別的資料夾（預設 data/hf）

已經下載過、大小和雜湊都對的檔案會略過。
English: download files listed in data/manifest.json from the Hugging Face dataset and verify SHA-256; no dependencies. Use --only to pick folders.
"""
import argparse, hashlib, json, shutil, sys, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(4 << 20), b''): h.update(c)
    return h.hexdigest()


def download(m, f, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{m['repo']}/resolve/main/{urllib.parse.quote(f['path'])}"
    with urllib.request.urlopen(url) as r, open(dst, 'wb') as out:
        while chunk := r.read(4 << 20): out.write(chunk)
    if sha256(dst) != f['sha256']:
        dst.unlink()
        sys.exit(f"雜湊不符：{f['path']}（檔案已刪除，重新執行會再下載）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=[])
    ap.add_argument('--out', default='data/hf')
    a = ap.parse_args()
    m = json.loads((ROOT / 'data/manifest.json').read_text())
    files = [f for f in m['files'] if not a.only or any(f['path'].startswith(p) for p in a.only)]
    total = sum(f['bytes'] for f in files)
    print(f"{len(files)} 個檔案，{total / 1e9:.2f} GB → {a.out}", flush=True)
    files.sort(key=lambda f: bool(f.get('same_as')))   # 內容重複的（same_as）最後從已下載的那份複製，不重複下載
    for i, f in enumerate(files, 1):
        dst = ROOT / a.out / f['path']
        if f.get('same_as'):
            src = ROOT / a.out / f['same_as']
            if not src.exists(): download(m, next(x for x in m['files'] if x['path'] == f['same_as']), src)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
            continue
        if dst.exists() and dst.stat().st_size == f['bytes'] and sha256(dst) == f['sha256']: continue
        print(f"[{i}/{len(files)}] {f['path']}", flush=True)
        download(m, f, dst)


if __name__ == '__main__':
    main()
