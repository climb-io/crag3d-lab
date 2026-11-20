"""戶外 3D Topo：把同一群另外建的一塊模型，用 COLMAP 的相機位置對齊到基準那塊。

Apple 建模會自己重新對位，所以同一群的照片分兩組建（例如岩壁上、岩壁下）時，兩塊各自只能靠 GPS 放到地圖上，
重疊又少，對不準。COLMAP 把整群照片對在同一個座標系：用基準那塊的相機求 COLMAP → 地圖的相似變換，
再把這塊的相機從 COLMAP 換到地圖上，求這塊目前位置 → 該在位置的相似變換，套到模型與相機。

用法：align_part.py <COLMAP 重建資料夾> <基準群資料夾> <要對齊的群資料夾>
改寫：<要對齊的群>/wall.raw.glb、wall.json（原本的存成 wall.geo.* 備份，重跑時從備份開始）
"""
import io, json, shutil, sys
from pathlib import Path
import numpy as np, pycolmap, trimesh
from PIL import Image

rec, base, part = pycolmap.Reconstruction(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
for f, g in (('wall.raw.glb', 'wall.geo.raw.glb'), ('wall.json', 'wall.geo.json')):
    if not (part / g).exists(): shutil.copyfile(part / f, part / g)
ctr = {im.name[:-4]: im.projection_center() for im in rec.images.values()}


def similarity(a, b):   # b ≈ s R a + t（Umeyama）
    ma, mb = a.mean(0), b.mean(0); A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (A ** 2).sum() * len(a)
    return s, R, mb - s * R @ ma


def fit(views):
    names = [v['name'] for v in views if v['name'] in ctr]
    a = np.array([ctr[n] for n in names]); b = np.array([next(v['at'] for v in views if v['name'] == n) for n in names])
    s, R, t = similarity(a, b); err = np.linalg.norm((s * a @ R.T + t) - b, axis=1)
    return (s, R, t), names, err


(s1, R1, t1), _, e1 = fit(json.loads((base / 'wall.json').read_text())['views'])
info = json.loads((part / 'wall.geo.json').read_text())
names = [v['name'] for v in info['views'] if v['name'] in ctr]
want = np.array([s1 * R1 @ ctr[n] + t1 for n in names])
have = np.array([next(v['at'] for v in info['views'] if v['name'] == n) for n in names])
s2, R2, t2 = similarity(have, want); e2 = np.linalg.norm(have * s2 @ R2.T + t2 - want, axis=1)
for _ in range(3):   # COLMAP 跟 Apple 對某些相機（例如岩壁上方往下拍的）算得不一致：殘差超過中位數兩倍的不用，再對一次
    ok = e2 <= max(0.05, 2 * np.median(e2))
    s2, R2, t2 = similarity(have[ok], want[ok]); e2 = np.linalg.norm(have * s2 @ R2.T + t2 - want, axis=1)
mesh = trimesh.load(part / 'wall.geo.raw.glb', force='mesh', process=False)
mesh.vertices = mesh.vertices * s2 @ R2.T + t2
tex = mesh.visual.material.baseColorTexture.convert('RGB'); buf = io.BytesIO(); tex.save(buf, 'JPEG', quality=85)   # trimesh 重存會變 PNG（4K 約 25 MB），改回 JPEG
mesh.visual.material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.open(io.BytesIO(buf.getvalue())), metallicFactor=0, roughnessFactor=1)
glb = mesh.export(file_type='glb')
(part / 'wall.raw.glb').write_bytes(glb)
for v in info['views']:
    v['at'] = (s2 * R2 @ np.array(v['at']) + t2).tolist(); v['rot'] = (R2 @ np.array(v['rot'])).tolist()
(part / 'wall.json').write_text(json.dumps(info))
move = np.linalg.norm(have - want, axis=1)
print(f"基準相機殘差中位數 {np.median(e1):.2f} m；這塊 {len(names)} 台：原本差 {np.median(move):.2f} m，對齊後殘差 {np.median(e2[ok]):.2f} m（用 {ok.sum()} 台，{len(ok) - ok.sum()} 台不一致不用），縮放 {s2:.3f}")
