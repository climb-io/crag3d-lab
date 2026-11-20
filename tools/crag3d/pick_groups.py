"""自動挑群：品質不夠的群直接丟，不硬合（不確定就丟）。
- 相機少於 15 台：太零碎，模型和定位都靠不住
- GPS 定位的中位誤差大於 2.5 m：放不準
- 朝下的面超過 30%：上下顛倒或傾倒（重力方向估錯）
- 直立的面少於 40%：拍到的多半是頂部或地面，不是岩壁面
- 跟別群在 GPS 位置上重疊超過 5%（0.4 m 內）：留相機多的那群
輸出 groups/keep.json：留下的群與每群丟掉的原因；groups/areas.json：留下的群依距離分區。"""
import itertools, json, sys
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree

D = Path(sys.argv[1])
info = {}
for p in sorted(D.glob('g*/wall.json')):
    r = json.loads(p.read_text()); g = p.parent.name
    info[g] = {'cameras': r['geo']['cameras'], 'geoErrM': round(r['geo']['medianErrM'], 2), 'center': r['center'], 'drop': None}
    if r['geo']['cameras'] < 15: info[g]['drop'] = '相機少於 15 台'
    elif r['geo']['medianErrM'] > 2.5: info[g]['drop'] = 'GPS 定位誤差太大'
    else:
        m = trimesh.load(p.parent / 'wall.raw.glb', force='mesh'); ny, a = m.face_normals[:, 1], m.area_faces
        down, upright = a[ny < -0.5].sum() / a.sum(), a[np.abs(ny) < 0.5].sum() / a.sum()
        info[g].update(down=round(float(down), 2), upright=round(float(upright), 2))
        if down > 0.3: info[g]['drop'] = f'方向錯（朝下的面 {down:.0%}）'
        elif upright < 0.4: info[g]['drop'] = f'沒拍到岩壁面（直立的面 {upright:.0%}）'
alive = [g for g in info if not info[g]['drop']]
pts = {g: trimesh.load(D / g / 'wall.raw.glb', force='mesh').sample(15000) for g in alive}
for a, b in sorted(itertools.combinations(alive, 2), key=lambda ab: -min(info[x]['cameras'] for x in ab)):
    if info[a]['drop'] or info[b]['drop']: continue
    lo, hi = sorted((a, b), key=lambda g: info[g]['cameras'])
    ov = float((cKDTree(pts[hi]).query(pts[lo])[0] < 0.4).mean())
    if ov > 0.05: info[lo]['drop'] = f'跟 {hi} 重疊 {ov:.0%}'
(D / 'keep.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
for g, r in info.items(): print(g, r['cameras'], r['geoErrM'], r['center'], r['drop'] or '留')

# 分區：留下的群在水平方向相距超過 5 m 就是不同區（中間沒有岩壁，不是洞），由西到東編號；每區各自一份模型、各自從岩壁下打開
keep = [g for g in info if not info[g]['drop']]
xz = {g: trimesh.load(D / g / 'wall.raw.glb', force='mesh').sample(8000)[:, [0, 2]] for g in keep}
parent = {g: g for g in keep}
def root(g):
    while parent[g] != g: g = parent[g]
    return g
for a, b in itertools.combinations(keep, 2):
    if cKDTree(xz[a]).query(xz[b])[0].min() < 5: parent[root(a)] = root(b)
areas = {}
for g in keep: areas.setdefault(root(g), []).append(g)
areas = sorted(areas.values(), key=lambda gs: min(xz[g][:, 0].min() for g in gs))
out = [{'area': i + 1, 'groups': sorted(gs), 'eastM': [round(float(min(xz[g][:, 0].min() for g in gs)), 1), round(float(max(xz[g][:, 0].max() for g in gs)), 1)]} for i, gs in enumerate(areas)]
(D / 'areas.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for a in out: print(f"區 {a['area']}：{'、'.join(a['groups'])}，東西 {a['eastM'][0]}～{a['eastM'][1]} m")
