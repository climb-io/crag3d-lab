"""戶外 3D Topo：各區的預設鏡頭和岩壁之間不能有東西擋著（例：飛鳥正前方有一根岩柱，鏡頭在它後面，畫面左邊一大塊糊的岩石）。

快照（snap.html）照路線範圍決定鏡頭；這裡用高斯潑濺模型裡不透明的點檢查：投到鏡頭畫面上，離鏡頭 CLOSE 公尺內（潑濺近看會糊成一大塊；遠一點的岩柱擋到一點是正常的岩場樣子）、比看的中心近 GAP 公尺以上、
落在畫面裡、路線最低點以上的點（鏡頭腳下的地面、草在那以下，不算），佔那塊畫面的格子比例超過 MAX_COVER 就算擋住；
離這一區的路線 ROUTE_NEAR 公尺內的不算（有路線的突出岩柱本來就要看到）。擋住的區試著繞看的中心左右轉（YAWS）、往前靠近（SCALES），
取改變最小、畫面乾淨的那個，寫進 crag.json 的 walls[].view = { yaw（度）, scale（離中心的距離倍數） }，snap.html 照它調鏡頭。

用法：clearance.py <模型.spz> <crag.json> <snaps.json> <輸出 crag.json>
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read
from scipy.spatial import cKDTree

SOLID, GAP, CLOSE, MAX_COVER, GRID, ROUTE_NEAR = 0.5, 2.5, 5.0, 0.01, 48, 1.5
YAWS = [0, 5, -5, 10, -10, 15, -15, 20, -20, 25, -25, 30, -30]
SCALES = [1, 0.85, 0.7]

spz, crag_file, snaps_file, out = sys.argv[1:]
s = read(spz); Q = s['xyz'][s['alpha'] >= SOLID * 255]
crag = json.loads(Path(crag_file).read_text())
snaps = {x['wall']: x for x in json.loads(Path(snaps_file).read_text())}


def cover(C, T, fov, aspect, P):
    """鏡頭在 C、看 T：比 T 近 GAP 以上、在畫面裡又高過路線點 P 最低點的點，佔那塊畫面格子的比例"""
    f = (T - C) / np.linalg.norm(T - C); r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    t = np.tan(np.radians(fov) / 2)
    proj = lambda X: ((X @ r) / (X @ f) / (t * aspect), (X @ u) / (X @ f) / t)
    px, py = proj(P - C)
    x0, x1, y0, y1 = -1, 1, max(py.min(), -1), 1
    X = Q - C; z = X @ f
    near = (z > 0.2) & (z < min(np.linalg.norm(T - C) - GAP, CLOSE))
    near[near] = cKDTree(P).query(Q[near], distance_upper_bound=ROUTE_NEAR)[0] > ROUTE_NEAR
    x, y = proj(X[near])
    ok = (x > x0) & (x < x1) & (y > y0) & (y < y1)
    cells = set(zip(((x[ok] - x0) / (x1 - x0) * GRID).astype(int), ((y[ok] - y0) / (y1 - y0) * GRID).astype(int)))
    return len(cells) / GRID ** 2


def move(C, T, yaw, scale):
    a = np.radians(yaw); d = C - T
    d = np.array([d[0] * np.cos(a) + d[2] * np.sin(a), d[1], -d[0] * np.sin(a) + d[2] * np.cos(a)])
    return T + d * scale


for w in crag['walls']:
    sn = snaps.get(w['name'])
    if not sn: continue
    C, T, fov, aspect = np.array(sn['camera']['position']), np.array(sn['camera']['target']), sn['camera']['fov'], sn['w'] / sn['h']
    P = np.array([p for r in crag['routes'] if w['routes'][0] <= int(r['number']) <= w['routes'][1] for p in r.get('points') or []])
    c0 = cover(C, T, fov, aspect, P)
    v0 = w.get('view') or {'yaw': 0, 'scale': 1}   # 快照已經照原本的設定拍（例：使用者指正的轉角、往前），這裡的調整疊加上去，不能清掉
    if c0 <= MAX_COVER: print(f"{w['name']}：沒有擋住（{c0:.1%}）"); continue
    best = next(((yaw, sc, c) for sc in SCALES for yaw in YAWS if (c := cover(move(C, T, yaw, sc), T, fov, aspect, P)) <= MAX_COVER), None)
    if best is None:
        best = min(((yaw, sc, cover(move(C, T, yaw, sc), T, fov, aspect, P)) for sc in SCALES for yaw in YAWS), key=lambda x: x[2])
    w['view'] = {'yaw': round(v0['yaw'] + best[0], 1), 'scale': round(v0.get('scale', 1) * best[1], 3)}
    print(f"{w['name']}：擋住 {c0:.1%} → 轉 {best[0]}°、距離 ×{best[1]}，剩 {best[2]:.1%}")
Path(out).write_text(json.dumps(crag, ensure_ascii=False))
