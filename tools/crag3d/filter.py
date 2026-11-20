"""戶外 3D Topo 實驗：只丟明顯沒救的照片（太糊、太暗或過曝、幾乎沒拍到岩石、連拍重複），其餘交給 3D 對位決定。

照片不是一張一張獨立的：拍的人站在一個點，上下、轉一圈把周圍拍得到的都拍，再走到下一個點。
照時間排，跟前一張隔超過 5 秒就是新的一站（station）。站內位置一樣，GPS 取全站中位數。
在岩壁上還是岩壁下以整站判斷：前後各 2 站（含自己）的站內俯仰中位數，再取中位數，往下超過 15° 就是岩壁上（top）。
岩壁上的站只用來補岩壁的上半部，對位時只跟附近的站配對（sfm.py）。"""
import json, sys
from datetime import datetime
from pathlib import Path
import cv2, numpy as np

W = Path(sys.argv[1])
rows = json.loads((W / 'prep.json').read_text())
med = float(np.median([r['sharp'] for r in rows]))


def rock(name):
    bgr = cv2.imread(str(W / 'img' / f'{name}.jpg'), cv2.IMREAD_REDUCED_COLOR_4)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1] / 255, hsv[..., 2] / 255
    sky = (v > 0.9) | ((h >= 95) & (h <= 130) & (s > 0.2) & (v > 0.5))
    green = (h >= 30) & (h <= 90) & (s > 0.25) & (v > 0.15)
    return float((~sky & ~green).mean())


def t(r): return datetime.strptime(r['time'], '%Y:%m:%d %H:%M:%S') if r.get('time') else None


order = sorted(rows, key=lambda r: (r.get('time') or '', r['name']))
stations = [[order[0]]]
for a, b in zip(order, order[1:]):
    (stations.append([b]) if not (t(a) and t(b)) or (t(b) - t(a)).total_seconds() > 5 else stations[-1].append(b))
pm = [float(np.median([r.get('pitch') or 0 for r in s])) for s in stations]
for k, s in enumerate(stations):
    top = float(np.median(pm[max(0, k - 2):k + 3])) > 15
    for r in s: r['station'], r['top'] = k, top
for r in rows:
    r['rock'] = rock(r['name'])
    r['drop'] = ('太糊' if r['sharp'] < 0.25 * med else
                 '太暗或過曝' if r['dark'] > 0.4 or r['blown'] > 0.4 else
                 '幾乎沒拍到岩石' if r['rock'] < 0.12 else None)
# 連拍重複：5 秒內、縮圖雜湊幾乎一樣，留比較清楚的那張
keep = [r for r in rows if not r['drop']]
for a, b in zip(keep, keep[1:]):
    if a['drop'] or not (t(a) and t(b)) or abs((t(b) - t(a)).total_seconds()) > 5: continue
    if sum(x != y for x, y in zip(a['dhash'], b['dhash'])) <= 12:
        (b if b['sharp'] < a['sharp'] else a)['drop'] = '跟前一張重複'
(W / 'filter.json').write_text(json.dumps([{k: r[k] for k in ('name', 'drop', 'station', 'top', 'sharp', 'rock')} for r in rows], ensure_ascii=False, indent=1))
from collections import Counter
print(Counter(r['drop'] for r in rows), f'{len(stations)} 站（岩壁上 {sum(s[0]["top"] for s in stations)} 站）', '岩壁上的照片（留下）', sum(r['top'] and not r['drop'] for r in rows))
print([(r['name'], r['drop']) for r in rows if r['drop']])
