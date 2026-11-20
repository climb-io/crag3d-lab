"""戶外 3D Topo 實驗：用路線順序淘汰配錯的投影。

路線編號依序排在岩壁上（大砲岩朝北，1 號在東，編號越大越西；61、62 號之間沒有岩壁）。
直接配對投上的路線裡混著配錯的（別處長得像的岩壁）。每一段裡挑出「位置照編號排」的最大集合，
基準路線一定要在裡面；不在這個集合裡的就是配錯，丟掉並記原因。

用法：order.py <投影 routes3d.json> <基準 routes3d.json> <輸出 json>
"""
import json, sys
import numpy as np

proj, anchor_file, out_file = sys.argv[1:4]
GAP, TOL = 61, 0.5   # 斷口；同一處的路線可以並排，容許往東退 0.5 m
anchors = {a['number'] for a in json.load(open(anchor_file)) if a.get('points')}
rows = json.load(open(proj))
east = lambda r: float(np.mean(np.array(r['points'])[:, 0]))

for side in (lambda n: n <= GAP, lambda n: n > GAP):
    cand = sorted((r for r in rows if r.get('points') and side(int(r['number']))), key=lambda r: int(r['number']))
    areas = {r['area'] for r in cand if r['number'] in anchors}
    cand = [r for r in cand if r['area'] in areas or r['number'] in anchors]   # 跟基準不同區的直接是配錯
    # 動態規劃：最長的「編號遞增、往西」序列，基準不能跳過
    n = len(cand); x = [east(r) for r in cand]; best = [None] * n
    for i in range(n):
        w = 1 + (1000 if cand[i]['number'] in anchors else 0)
        best[i] = (w, None)
        for j in range(i):
            if x[i] <= x[j] + TOL and best[j][0] + w > best[i][0]:
                skipped = any(cand[k]['number'] in anchors for k in range(j + 1, i))
                if not skipped: best[i] = (best[j][0] + w, j)
    i = max(range(n), key=lambda k: best[k][0]) if n else None
    keep = set()
    while i is not None: keep.add(cand[i]['number']); i = best[i][1]
    for r in rows:
        if r.get('points') and side(int(r['number'])) and r['number'] not in keep:
            r['drop'] = f"位置不合路線順序（往東 {east(r):.1f} m）"; r['area'] = None; del r['points']

json.dump(rows, open(out_file, 'w'), ensure_ascii=False, indent=1)
ok = [r for r in rows if r.get('points')]
print('投上', len(ok), '條：', [r['number'] for r in ok])
from collections import Counter
print(Counter(r['drop'].split('（')[0] for r in rows if 'drop' in r))
