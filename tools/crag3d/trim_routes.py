"""戶外 3D Topo：路線頭尾離岩面太遠的點刪掉。

路線是照片上的線投到 3D 的，靠近天際線、岩頂的那段常常投到岩面外（浮在空中），換個角度看就像畫在天空裡。
從頭、從尾一路往內，離模型表面超過 OFF 公尺的點刪掉，碰到貼著岩面的點就停（中間的不動）。剩不到 MIN_PTS 點的路線不改。

用法：trim_routes.py <crag_pack 的 crag.json> <輸出 crag.json> <模型 wall.edge.raw.glb>…
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree

OFF, MIN_PTS = 0.35, 4
src, dst, parts = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
mesh = trimesh.util.concatenate([trimesh.load(p, force='mesh', process=False) for p in parts])
surf = cKDTree(trimesh.sample.sample_surface_even(mesh, 400000, seed=0)[0])
pack = json.loads(src.read_text())
for r in pack['routes']:
    p = np.array(r['points']); d = surf.query(p)[0]
    on = np.flatnonzero(d <= OFF)
    if len(on) < MIN_PTS: continue
    a, b = on[0], on[-1]
    if a or b < len(p) - 1:
        print(f"{r['number']}：頭刪 {a} 點、尾刪 {len(p) - 1 - b} 點（尾端原本離岩面 {d[-1]:.2f} m）")
    r['points'] = p[a:b + 1].tolist()
dst.write_text(json.dumps(pack, ensure_ascii=False, separators=(',', ':')))
