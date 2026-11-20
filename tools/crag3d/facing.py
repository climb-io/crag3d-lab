"""戶外 3D Topo：每面岩壁的朝向（岩面的水平法線，朝向起攀的地方），寫進 crag.json 的 walls[].facing，快照與紅點貓 3D 的各區鏡頭正對岩面。

照片的站位（standAt）常常偏在一側，用它推朝向，打開的角度會斜斜的。改用高斯潑濺模型裡路線經過的岩面：
每個路線點附近 RADIUS 公尺的岩面點做主成分分析，最小的那一軸是那裡的法線（朝向 standAt 那一側），整面岩壁取平均的水平方向。
只看路線經過的地方，旁邊轉角的另一面岩壁不會算進去。

用法：facing.py <模型.spz> <crag.json> <輸出 crag.json>
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read
from scipy.spatial import cKDTree

RADIUS, SOLID = 0.6, 0.5
spz, src, out = sys.argv[1:]
s = read(spz); q = s['xyz'][s['alpha'] >= SOLID * 255]; tree = cKDTree(q)
crag = json.loads(Path(src).read_text())
for w in crag['walls']:
    pts = np.array([p for r in crag['routes'] if w['routes'][0] <= int(r['number']) <= w['routes'][1] for p in r['points']])
    toward = np.array(w['standAt'])[[0, 2]] - np.array(w['center'])[[0, 2]]
    ns = []
    for near in tree.query_ball_point(pts, RADIUS):
        if len(near) < 20: continue
        x = q[near] - q[near].mean(0)
        n3 = np.linalg.svd(x, full_matrices=False)[2][2]
        h = n3[[0, 2]]
        if np.linalg.norm(h) < 0.3: continue   # 幾乎水平的岩面（岩階頂、地面）不算
        h = h / np.linalg.norm(h)
        ns.append(h if h @ toward > 0 else -h)
    n = np.mean(ns, 0); n /= np.linalg.norm(n); m = len(ns)
    old = toward / np.linalg.norm(toward)
    w['facing'] = [round(float(n[0]), 4), 0, round(float(n[1]), 4)]
    print(f"{w['name']}：{m} 個路線點，跟站位的方向差 {np.degrees(np.arccos(np.clip(n @ old, -1, 1))):.0f}°")
Path(out).write_text(json.dumps(crag, ensure_ascii=False, indent=1))
