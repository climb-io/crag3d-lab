"""戶外 3D Topo：照岩面涵蓋度重新挑一面岩壁（例：砲管三個面）要建模的照片，舊照片、補拍的照片一起挑。

1. 相機位置：COLMAP 重建（舊照片＋補拍）用已在地圖上的舊照片（基準群 wall.json 的 views）求 COLMAP → 地圖的相似變換
   （殘差超過中位數兩倍的不用、重對三次，同 align_part.py），所有註冊到的照片都換到地圖上；沒註冊到的判斷不了，不挑。
2. 岩面：參考網格（同一套地圖座標）在岩壁範圍（左右前後上 MARGIN 公尺、往下 BELOW 公尺）裡取點與法線。
3. 每張照片對每個點：在畫面內、在相機前方、距離 MAX_DIST 公尺內、視線跟法線夾角 MAX_ANGLE 度內才算看得到；
   品質 = 每公尺像素（焦距 ÷ 距離）× 正對程度（cos）。不算遮擋（凸出的岩塊靠法線分得出面）。
4. 貪婪挑選：每次挑讓「每個點前 K（12）好的視角品質總和」增加最多的照片，直到 MAX 張或增加不到 MIN_GAIN。
   遠一點的照片一張涵蓋整面、近的細節清楚，兩種會自然混在一起。

用法：select_cover.py <COLMAP 重建> <基準群 wall.json> <參考網格 .glb> <crag.json> <岩壁名稱> <prep.json…> --out=<輸出 json> [--max=260]
輸出：{photos: 照片名（照拍攝順序）, cover: 各面涵蓋統計, views: 地圖上的相機}
"""
import json, sys
from pathlib import Path
import numpy as np, pycolmap, trimesh

MARGIN, BELOW, MAX_DIST, MAX_ANGLE, K, MIN_GAIN, N_PTS = 2.0, 3.0, 45.0, 75.0, 12, 0.002, 6000
args = [a for a in sys.argv[1:] if not a.startswith('--')]
opt = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--'))
rec_dir, base_json, ref_glb, crag_json, name, *preps = args
MAX = int(opt.get('max', 260))

rec = pycolmap.Reconstruction(rec_dir)
base = {v['name']: v for v in json.loads(Path(base_json).read_text())['views']}
meta = {r['name']: r for p in preps for r in json.loads(Path(p).read_text())}
later = {r['name'] for p in preps[1:] for r in json.loads(Path(p).read_text())}   # 第二個以後的 prep.json 是補拍


def similarity(a, b):   # b ≈ s R a + t（Umeyama）
    ma, mb = a.mean(0), b.mean(0); A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (A ** 2).sum() * len(a)
    return s, R, mb - s * R @ ma


ims = {im.name[:-4]: im for im in rec.images.values()}
common = [n for n in ims if n in base]
a = np.array([ims[n].projection_center() for n in common]); b = np.array([base[n]['at'] for n in common])
use = np.ones(len(common), bool)
for _ in range(3):
    s, R, t = similarity(a[use], b[use]); err = np.linalg.norm((s * a @ R.T + t) - b, axis=1)
    use = err <= 2 * np.median(err)
print(f'對齊：共同 {len(common)} 張、用 {use.sum()} 張，殘差中位 {np.median(err[use]):.3f} m')

views = {}
for n, im in ims.items():
    m = meta.get(n)
    if not m: continue
    Rwc = im.cam_from_world().rotation.matrix().T   # 欄：相機的 x 右、y 下、z 前（COLMAP 世界座標）
    views[n] = dict(at=(s * R @ im.projection_center() + t).tolist(), rot=(R @ Rwc @ np.diag([1, -1, -1])).tolist())   # 地圖：x 右、y 上、-z 前

crag = json.loads(Path(crag_json).read_text()); wall = next(w for w in crag['walls'] if w['name'] == name)
lo, hi = np.array(wall['min'], float) - [MARGIN, BELOW, MARGIN], np.array(wall['max'], float) + MARGIN
mesh = trimesh.load(ref_glb, force='mesh')
inside = np.all((mesh.triangles_center >= lo) & (mesh.triangles_center <= hi), axis=1)
sub = mesh.submesh([np.where(inside)[0]], append=True)
P, fi = trimesh.sample.sample_surface_even(sub, N_PTS, seed=0); Nrm = sub.face_normals[fi]
print(f'岩面取點 {len(P)}（{sub.area:.0f} m²）')

names = sorted(views); Q = np.zeros((len(names), len(P)), np.float32)
cosmax = np.cos(np.radians(MAX_ANGLE))
for i, n in enumerate(names):
    m, v = meta[n], views[n]; C, Rm = np.array(v['at']), np.array(v['rot'])
    foc = (m.get('focal35') or 26) / 43.27 * np.hypot(m['w'], m['h'])   # 以 prep.json 的縮圖像素計；35 mm 等效焦距的對角線 43.27 mm
    X = (P - C) @ Rm; z = -X[:, 2]; d = np.linalg.norm(P - C, axis=1)
    ray = (C - P) / d[:, None]; cs = (ray * Nrm).sum(1)
    ok = (z > 0.3) & (d < MAX_DIST) & (cs > cosmax)
    x, y = foc * X[:, 0] / np.maximum(z, 1e-3), foc * X[:, 1] / np.maximum(z, 1e-3)
    ok &= (np.abs(x) < m['w'] / 2 * 0.95) & (np.abs(y) < m['h'] / 2 * 0.95)
    Q[i] = np.where(ok, foc / d * cs, 0)
Q /= max(Q.max(), 1e-6)

top = np.zeros((K, len(P)), np.float32)   # 每點目前前 K 好的品質（由大到小）
def total(t): return float(t.sum())
picked, cur = [], 0.0
avail = np.ones(len(names), bool); avail[Q.max(1) == 0] = False
while len(picked) < MAX and avail.any():
    gains = np.full(len(names), -1.0)
    for i in np.where(avail)[0]:
        q = Q[i]; better = q > top[-1]
        if not better.any(): avail[i] = False; continue
        # 把 q 插進前 K：增益 = 新的前 K 總和 − 舊的
        t = np.vstack([top[:, better], q[better][None]]); t = -np.sort(-t, axis=0)[:K]
        gains[i] = t.sum() - top[:, better].sum()
    i = int(np.argmax(gains))
    if gains[i] < MIN_GAIN * len(P): break
    q = Q[i]; t = np.vstack([top, q[None]]); top = -np.sort(-t, axis=0)[:K]
    picked.append(names[i]); avail[i] = False

seen = (Q[[names.index(n) for n in picked]] > 0)
cnt = seen.sum(0)
# 各面：法線水平方向分三段（左側、正面、右側），用岩壁的 facing（沒有就站位方向）當正面
f = np.array(wall.get('facing') or (np.array(wall['standAt']) - np.array(wall['center'])), float); f[1] = 0; f /= np.linalg.norm(f)
h = Nrm.copy(); h[:, 1] = 0; ang = np.degrees(np.arctan2(np.cross(f, h)[:, 1], (h * f).sum(1)))
faces = {'左側': ang > 45, '正面': np.abs(ang) <= 45, '右側': ang < -45, '頂部': Nrm[:, 1] > 0.7}
cover = {k: dict(points=int(m_.sum()), none=round(float((cnt[m_] == 0).mean()), 3) if m_.any() else None,
                 median=int(np.median(cnt[m_])) if m_.any() else None) for k, m_ in faces.items()}
new = [n for n in picked if n in later]
print(f'挑 {len(picked)} 張（補拍 {len(new)}、舊 {len(picked) - len(new)}）；各面點數／沒照片看到的比例／每點張數中位：', cover)
Path(opt['out']).write_text(json.dumps(dict(photos=sorted(picked), new=sorted(new), cover=cover, views={n: views[n] for n in picked}), ensure_ascii=False))
