"""戶外 3D Topo 實驗：以已經投上 3D 的路線為基準，依路線順序往兩邊推。

路線照編號依序排在岩壁上，相鄰路線的舊照片常是同一面牆（甚至同一張底圖畫不同的線），
同一年同一台相機拍，彼此很好配對。基準照片的拍攝位置已知，它的特徵點打到模型上就是 3D 點；
相鄰路線的照片跟基準照片配對，借這些 3D 點用 PnP 求自己的拍攝位置，再把紅線打到模型上。
投上的路線變成新的基準，一路往兩邊推；每條只找編號差 4 以內、同一段（不跨過斷口）的基準。

用法：propagate.py <工作資料夾> <全部路線 json> <基準 routes3d.json> <輸出資料夾>
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, pycolmap, trimesh

D, routes_file, anchor_file, OUT = map(Path, sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
GAP = 61   # 61 和 62 號之間沒有岩壁（舊區域圖），不跨過去推
side = lambda n: int(n) <= GAP

sys.path.insert(0, str(Path(__file__).parent)); from redline import trace, wall_run   # 同一資料夾的共用模組
areas = json.loads((D / 'groups/areas.json').read_text())
meshes = {}
for a in areas:
    m = trimesh.util.concatenate([trimesh.load(D / 'groups' / g / 'wall.raw.glb', force='mesh') for g in a['groups']])
    meshes[a['area']] = trimesh.ray.ray_triangle.RayMeshIntersector(m)


def hit(area, C, d):
    out = np.full((len(d), 3), np.nan)
    loc, idx, _ = meshes[area].intersects_location(np.broadcast_to(C, d.shape), d, multiple_hits=False)
    out[idx] = loc
    return out


def rays(K, rv, tv, uv):
    R = cv2.Rodrigues(np.asarray(rv, float))[0]
    C = (-R.T @ np.asarray(tv, float).reshape(3)).ravel()
    d = (np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(np.asarray(K)).T) @ R
    return C, d / np.linalg.norm(d, axis=1, keepdims=True)


routes = {r['number']: r for r in json.loads(routes_file.read_text()) if r['number'].isdigit()}
sift = cv2.SIFT_create(nfeatures=5000)
feats = {}
for n, r in routes.items():
    img = cv2.imread(str(routes_file.parent / r['file']))
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    feats[n] = (np.float32([k.pt for k in kp]), des, img)

placed = {}
for a in json.loads(anchor_file.read_text()):
    if not a.get('points'): continue
    a['via'] = None; placed[a['number']] = a
results = {}
bf = cv2.BFMatcher(cv2.NORM_L2)
anchor3d, tried = {}, set()   # 基準照片特徵點的 3D 點只算一次；試過的（路線、基準）不再試


def points3d(p):
    if p not in anchor3d:
        A = placed[p]; C, d = rays(A['pose']['K'], A['pose']['rvec'], A['pose']['tvec'], feats[p][0])
        anchor3d[p] = hit(A['area'], C, d)
    return anchor3d[p]


changed = True
while changed:
    changed = False
    todo = sorted((n for n in routes if n not in placed), key=lambda n: min([abs(int(n) - int(p)) for p in placed if side(p) == side(n)] or [99]))
    for n in todo:
        cands = sorted((p for p in placed if side(p) == side(n) and abs(int(p) - int(n)) <= 4), key=lambda p: abs(int(p) - int(n)))
        if not cands: continue
        r = routes[n]; pts, des, img = feats[n]
        line = trace(img)
        if line is None: results[n] = {'drop': '描不到紅線'}; continue
        for p in cands:
            if (n, p) in tried: continue
            tried.add((n, p))
            A = placed[p]; da = feats[p][1]; Xa = points3d(p)
            ms = [m for m, k in bf.knnMatch(des, da, k=2) if m.distance < 0.8 * k.distance and not np.isnan(Xa[m.trainIdx, 0])]
            if len(ms) < 20: results[n] = {'drop': f'跟基準 {p} 號配對太少', 'via': p}; continue
            P2 = pts[[m.queryIdx for m in ms]]; P3 = Xa[[m.trainIdx for m in ms]]
            h, w = img.shape[:2]; best = None
            fa = A['pose']['K'][0][0] / max(feats[p][2].shape[:2])   # 舊照片多半同一台相機：沿用基準的焦距，只在附近微調
            for fk in fa * np.array([0.85, 0.93, 1, 1.08, 1.17]):
                K = np.array([[fk * max(w, h), 0, w / 2], [0, fk * max(w, h), h / 2], [0, 0, 1]])
                ok, rv, tv, inl = cv2.solvePnPRansac(P3, P2, K, None, reprojectionError=4, iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
                if ok and inl is not None and (best is None or len(inl) > best[0]): best = (len(inl), K, rv, tv)
            if best is None or best[0] < 20: results[n] = {'drop': f'借基準 {p} 號求不出拍攝位置', 'via': p}; continue
            ni, K, rv, tv = best
            C, d = rays(K, rv.ravel(), tv.ravel(), line)
            X, ratio = wall_run(hit(A['area'], C, d))
            if ratio < 0.6: results[n] = {'drop': f'紅線連續打到模型的只有 {ratio:.0%}（超出模型範圍）', 'via': p}; continue
            placed[n] = {**{k: r[k] for k in ('id', 'number', 'name', 'grade', 'color')}, 'area': A['area'], 'points': X.round(3).tolist(),
                         'hitRatio': round(ratio, 2), 'pnpInliers': int(ni), 'via': p, 'camera': C.round(2).tolist(),
                         'pose': {'K': K.round(2).tolist(), 'rvec': rv.ravel().tolist(), 'tvec': tv.ravel().tolist()}}
            results.pop(n, None); changed = True
            print(f"{n}. {r['name']} ← 基準 {p} 號，內點 {ni}，打到 {ratio:.0%}", flush=True)
            break

out = []
for n, r in sorted(routes.items(), key=lambda kv: int(kv[0])):
    if n in placed: out.append(placed[n])
    else: out.append({**{k: r[k] for k in ('id', 'number', 'name', 'grade', 'color')}, **results.get(n, {'drop': '附近沒有基準'})})
(OUT / 'routes3d.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
from collections import Counter
print(Counter('投上了' if 'points' in x else x['drop'].split('（')[0].split(' ')[0] for x in out))
# 順序檢查：同一區投上的路線，沿岩壁的位置應該照編號排（大砲岩朝北，編號由東往西）
for a in areas:
    xs = [(int(x['number']), float(np.mean(np.array(x['points'])[:, 0]))) for x in out if x.get('area') == a['area']]
    xs.sort(); bad = sum(1 for (_, e1), (_, e2) in zip(xs, xs[1:]) if e2 > e1 + 0.3)
    print(f"區 {a['area']}：{[n for n, _ in xs]}，東西順序不合的相鄰對 {bad}")
