"""戶外 3D Topo：挑一面岩壁要建模的照片（岩壁下、岩壁上方往下拍的都算），照片完全不過濾品質。

岩壁下（filter.py 判斷的站）：相機位置與朝向用整個岩場先建一次的各群（build.sh 的 groups/g*/wall.json，相機已放到地圖上）。
焦距用照片的 35 mm 等效焦距（prep.json 的 focal35），畫面長寬用 prep.json 轉正後的 w、h。
岩面範圍（crag.json 的 walls：min、max）左右、前後、往上各放寬 MARGIN 公尺、往下放寬 BELOW 公尺（岩壁腳和地面），在範圍裡取點；
相機離岩面中心水平 REACH 公尺內，而且範圍裡至少 FRAC 的點落在畫面內、在相機前方，就挑進來。
再以站為單位補齊：站裡有一張挑進來，同站沒有相機位置（判斷不了）的照片也挑進來。
岩壁上方：小群的相機位置靠 GPS、不準，改用整站：站的 GPS 中位數離岩面中心水平 TOP_REACH 公尺內，整站挑進來（往下拍岩頂的照片）。
照片太多 Apple 會用光記憶體（8 GB 的 Mac 約 280 張），所以岩壁上方只取近的站；總數超過 MAX，從最遠的岩壁上方站開始拿掉。

用法：select_wall.py <groups 資料夾> <prep.json> <filter.json> <crag.json> <岩壁名稱> <輸出 json>
輸出：照片檔名（不含副檔名），照檔名（拍攝順序）排。
"""
import json, sys
from pathlib import Path
import numpy as np

MARGIN, BELOW, REACH, TOP_REACH, FRAC, MAX = 2.0, 3.0, 15.0, 10.0, 0.05, 280
ORIGIN, EARTH = (121.5199, 25.1424), 6378137.0   # clean.py 的岩場原點
groups, prep, filt, crag, name, out = Path(sys.argv[1]), *(json.loads(Path(a).read_text()) for a in sys.argv[2:5]), sys.argv[5], Path(sys.argv[6])
wall = next(w for w in crag['walls'] if w['name'] == name)
meta = {r['name']: r for r in prep}
station = {r['name']: r['station'] for r in filt}
bottom = {r['name'] for r in filt if not r['top']}
views = {v['name']: v for g in sorted(groups.glob('g*/wall.json')) for v in json.loads(g.read_text())['views']}

lo, hi = np.array(wall['min'], float), np.array(wall['max'], float)
c = (lo + hi) / 2
grid = np.meshgrid(*[np.linspace(lo[i] - (BELOW if i == 1 else MARGIN), hi[i] + MARGIN, 6) for i in range(3)], indexing='ij')
P = np.stack([g.ravel() for g in grid], 1)
picked = []
for k, v in views.items():
    m = meta.get(k)
    C, R = np.array(v['at']), np.array(v['rot'])
    if not m or k not in bottom or np.linalg.norm((C - c)[[0, 2]]) > REACH: continue
    foc = (m.get('focal35') or 26) / 43.27 * np.hypot(m['w'], m['h'])   # 35 mm 等效焦距的對角線是 43.27 mm
    X = (P - C) @ R   # 相機座標：x 右、y 上、-z 前
    z = -X[:, 2]; front = z > 0.3
    x, y = foc * X[front, 0] / z[front], foc * X[front, 1] / z[front]
    if ((np.abs(x) < m['w'] / 2) & (np.abs(y) < m['h'] / 2)).sum() >= FRAC * len(P): picked.append(k)

keep = {station[p] for p in picked}
gps = {}
for r in prep:
    if r['name'] in station and r['name'] not in bottom and 'lat' in r:
        gps.setdefault(station[r['name']], []).append([np.radians(r['lng'] - ORIGIN[0]) * EARTH * np.cos(np.radians(ORIGIN[1])), -np.radians(r['lat'] - ORIGIN[1]) * EARTH])
dist = {st: np.linalg.norm(np.median(xz, 0) - c[[0, 2]]) for st, xz in gps.items()}
top = sorted((st for st, d in dist.items() if d <= TOP_REACH), key=dist.get)
base = set(picked) | {p for p, st in station.items() if st in keep and p not in views}
size = {st: sum(1 for p, s2 in station.items() if s2 == st and p not in base) for st in top}
while top and len(base) + sum(size[st] for st in top) > MAX: top.pop()
top = set(top)
full = sorted(base | {p for p, st in station.items() if st in top})
out.write_text(json.dumps(full, indent=0))
print(f'{name}：岩壁下看得到岩面 {len(picked)} 張（{len(keep)} 站）、岩壁上方 {len(top)} 站，共 {len(full)} 張（{full[0]}～{full[-1]}）')
