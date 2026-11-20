"""戶外 3D Topo 實驗：岩面（例如梁大隆）在 3D 裡的範圍與從岩壁下看它的位置。

岩面的定義是一段路線號碼（來自區域圖）；範圍就是這些路線投到 3D 的點。
岩面朝哪裡：路線底下模型表面的法向量（水平分量）平均；路線常在岩柱、轉角的某一側，只看路線中心會從側面看，路線像畫在天空。
從岩壁下看的位置取建模照片裡站在岩面正前方（偏離朝向 FACE_DEG 度以內）、朝著岩面拍、離岩面最近的那一張（真的有人站過的地方）；
正前方沒有照片就放寬到所有朝著岩面拍的照片，取最正面的。
岩面裡一條路線都沒投上就定不出來，記原因。

用法：walls.py <工作資料夾> <岩面定義 json> <routes3d.json> <輸出 json>
"""
import json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree

FACE_DEG = 30

D, walls_file, routes_file, out_file = map(Path, sys.argv[1:5])
routes = {int(r['number']): r for r in json.loads(routes_file.read_text())}
areas = json.loads((D / 'groups/areas.json').read_text())
views = [{**v, 'area': a['area']} for a in areas for g in a['groups'] for v in json.loads((D / 'groups' / g / 'wall.json').read_text())['views']]
meshes = {a['area']: trimesh.util.concatenate([trimesh.load(D / 'groups' / g / 'wall.raw.glb', force='mesh', process=False) for g in a['groups']]) for a in areas}

out = []
for w in json.loads(walls_file.read_text()):
    lo, hi = w['routes']
    placed = [routes[n] for n in range(lo, hi + 1) if routes.get(n, {}).get('points')]
    row = {'name': w['name'], 'routes': [lo, hi], 'placed': [int(r['number']) for r in placed]}
    if not placed: out.append({**row, 'drop': '沒有路線投上 3D'}); continue
    P = np.concatenate([np.array(r['points']) for r in placed])
    ctr = P.mean(0); area = placed[0]['area']
    m = meshes[area]; _, fi = cKDTree(m.triangles_center).query(P, k=8)
    n = m.face_normals[fi].mean(1)
    cams = np.array([v['at'] for v in views if v['area'] == area])
    n *= np.sign(((cams[cKDTree(cams).query(P)[1]] - P) * n).sum(1))[:, None]   # 模型的面方向不一定一致：翻到拍照的那一側
    face = n.mean(0); face[1] = 0; face /= np.linalg.norm(face)
    best = None
    for strict in (True, False):
        for v in views:
            if v['area'] != area: continue
            to = ctr - np.array(v['at']); dist = np.linalg.norm(to); flat = -to.copy(); flat[1] = 0
            if np.dot(v['dir'], to / dist) < 0.8: continue   # 要朝著岩面拍
            front = np.dot(flat / np.linalg.norm(flat), face)
            if strict and front < np.cos(np.radians(FACE_DEG)): continue
            key = dist if strict else -front
            if best is None or key < best[0]: best = (key, v)
        if best: break
    row.update(area=area, center=ctr.round(2).tolist(), top=[round(float(ctr[0]), 2), round(float(P[:, 1].max()) + 1.2, 2), round(float(ctr[2]), 2)],
               min=P.min(0).round(2).tolist(), max=P.max(0).round(2).tolist(), facing=face.round(3).tolist())
    if best: row.update(standAt=best[1]['at'], standPhoto=best[1]['name'], standDistM=round(float(best[0]), 1))
    out.append(row)
out_file.write_text(json.dumps(out, ensure_ascii=False, indent=1))
for w in out:
    where = w['drop'] if 'drop' in w else f"東西 {w['min'][0]}～{w['max'][0]} m，站在 {w.get('standPhoto')}（{w.get('standDistM')} m）"
    print(w['name'], f"{w['routes'][0]}–{w['routes'][1]}", '投上', w['placed'], where)
