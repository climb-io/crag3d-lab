"""戶外 3D Topo 實驗：自動修模型、放到地圖上、壓成小 GLB。只刪不改（保留貼圖座標），不需要人調。

用法：clean.py <建模輸出資料夾（OBJ＋apple-poses.json 在上一層）> <prep.json> <輸出.glb> [舊模型.raw.glb]
舊模型：同一群加照片重建時，新模型對齊到舊模型（不重新對 GPS），位置不會跳。
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, trimesh
from PIL import Image

src, prep, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
anchor = Path(sys.argv[4]) if len(sys.argv) > 4 else None
obj = next(src.glob('*.obj'))
mesh = trimesh.load(obj, process=False, force='mesh')
report = {'facesIn': len(mesh.faces)}


def keep_faces(m, mask):
    m.update_faces(mask); m.remove_unreferenced_vertices(); return m


# 1. 細長的破三角形與跨過空洞的大面：邊長超過中位數 6 倍的面
e = mesh.edges_unique_length
med = float(np.median(e))
long_edge = np.zeros(len(mesh.edges_unique), bool); long_edge[e > 6 * med] = True
bad = long_edge[mesh.faces_unique_edges].any(axis=1)
mesh = keep_faces(mesh, ~bad); report['longEdgeFaces'] = int(bad.sum())

# 貼圖接縫的頂點在 OBJ 裡是分開的兩份：連通用「位置相同算同一點」判斷
def merged(m):
    _, inv = np.unique(np.round(m.vertices, 6), axis=0, return_inverse=True)
    return inv.reshape(-1)[m.faces]



# 2. 孤立的小碎塊：小於最大塊 5% 的連通區塊丟掉
f = merged(mesh)
adj = trimesh.graph.face_adjacency(faces=f)
labels = trimesh.graph.connected_component_labels(adj, node_count=len(f))
sizes = np.bincount(labels)
mesh = keep_faces(mesh, sizes[labels] >= 0.05 * sizes.max())
report['components'] = sorted(sizes[sizes >= 0.05 * sizes.max()].tolist(), reverse=True)

# 3. 放到地圖上：建模座標的相機位置對 GPS（東、北），只求水平的旋轉、縮放、平移；高度不用 GPS（矮牆的 GPS 高度太亂），
#    建模座標本來就是重力向上（y 軸），比例用水平距離估。外點多（GPS 誤差 3–16 m），用 RANSAC。
poses = json.loads((src.parent / 'apple-poses.json').read_text())

# 轉正：模型的「上」由 Object Capture 估，常歪十幾度、小群甚至顛倒。每張照片的重力感應器給相機俯仰 p（往下為正，pitch.py），
# 模型裡的相機前方 f 跟真正的上方 U 滿足 f·U = -sin(p)；所有相機一起解最小平方的 U，再把 U 轉到 +y。
def fwd(q):
    x, y, z, w = q
    return np.array([-(2 * (x * z + w * y)), -(2 * (y * z - w * x)), -(1 - 2 * (x * x + y * y))])   # 相機看 -Z


pitch = {r['name']: r.get('pitch') for r in json.loads(prep.read_text())}
F = np.array([fwd(p['quaternion']) for p in poses if pitch.get(Path(p['name']).stem) is not None])
C = np.array([-np.sin(np.radians(pitch[Path(p['name']).stem])) for p in poses if pitch.get(Path(p['name']).stem) is not None])
U = np.linalg.lstsq(F, C, rcond=None)[0]; U /= np.linalg.norm(U)
ax = np.cross(U, [0, 1, 0]); sn, cs = np.linalg.norm(ax), U[1]
K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]) / (sn or 1)
Rg = np.eye(3) + sn * K + (1 - cs) * K @ K
res = np.degrees(np.arcsin(np.clip(-(F @ Rg.T)[:, 1], -1, 1))) - np.degrees(np.arcsin(np.clip(-C, -1, 1)))
report['tiltDeg'] = round(float(np.degrees(np.arccos(np.clip(cs, -1, 1)))), 1); report['pitchResDeg'] = round(float(np.median(np.abs(res))), 1)
for p in poses: p['fwd'] = Rg @ fwd(p['quaternion']); p['position'] = (Rg @ np.array(p['position'])).tolist()
mesh.vertices = mesh.vertices @ Rg.T
gps = {r['name']: r for r in json.loads(prep.read_text()) if 'lat' in r}
_all = [r for r in json.loads(prep.read_text()) if 'lat' in r]
lng0, lat0, alt0 = ORIGIN = (round(float(np.mean([r['lng'] for r in _all])), 4), round(float(np.mean([r['lat'] for r in _all])), 4), round(float(np.median([r['alt'] for r in _all]))))   # 整個岩場共用的原點（所有照片 GPS 的平均），各群放進同一個座標系
R = 6378137.0
A, B, W, H, Y = [], [], [], [], []
for p in poses:
    g = gps.get(Path(p['name']).stem)
    if not g: continue
    A.append([p['position'][0], -p['position'][2]])   # 建模座標：x 右、y 上、z 朝外 → 水平面 (x, -z)
    B.append([np.radians(g['lng'] - lng0) * R * np.cos(np.radians(lat0)), np.radians(g['lat'] - lat0) * R])
    W.append(1 / max(g.get('accuracy') or 10, 1))
    H.append(g['alt'] - alt0); Y.append(p['position'][1])
A, B, W, H, Y = np.array(A), np.array(B), np.array(W), np.array(H), np.array(Y)


def sim2(a, b, w):
    ca, cb = (a * w[:, None]).sum(0) / w.sum(), (b * w[:, None]).sum(0) / w.sum()
    x, y = a - ca, b - cb
    H = (x * w[:, None]).T @ y
    U, S, Vt = np.linalg.svd(H); D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T; s = (S * np.diag(D)).sum() / (w * (x ** 2).sum(1)).sum()
    return s, Rm, cb - s * Rm @ ca


rng = np.random.default_rng(0); best = None
for _ in range(500):
    i = rng.choice(len(A), 2, replace=False)
    if np.linalg.norm(A[i[0]] - A[i[1]]) < 1e-6: continue
    s, Rm, t = sim2(A[i], B[i], np.ones(2))
    err = np.linalg.norm((s * (Rm @ A.T)).T + t - B, axis=1)
    inl = err < 8
    if best is None or inl.sum() > best.sum(): best = inl
s, Rm, t = sim2(A[best], B[best], W[best])
err = np.linalg.norm((s * (Rm @ A.T)).T + t - B, axis=1)
report['geo'] = {'cameras': len(A), 'inliers': int(best.sum()), 'scale': float(s), 'medianErrM': float(np.median(err[best])),
                 'rotationDeg': float(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0])))}
report['sizeM'] = (mesh.extents * s).round(1).tolist()

# 4. 貼圖照建模時設定的大小（不再縮）、存 JPEG；交給 gltfpack 做 meshopt 壓縮、量化
tex = mesh.visual.material.baseColorTexture if hasattr(mesh.visual.material, 'baseColorTexture') else mesh.visual.material.image
tex = tex.convert('RGB')
import io; buf = io.BytesIO(); tex.save(buf, 'JPEG', quality=80); tex = Image.open(io.BytesIO(buf.getvalue()))   # GLB 裡存 JPEG，不是 PNG
mesh.visual.material = trimesh.visual.material.PBRMaterial(baseColorTexture=tex, metallicFactor=0, roughnessFactor=1)
# 頂點換成岩場座標（公尺）：x 東、y 上、z 南（three.js 慣例）；高度取相機 GPS 高度的中位數差
up = float(np.median(H - s * Y))
v = mesh.vertices; e = (s * (Rm @ np.stack([v[:, 0], -v[:, 2]]))).T + t
mesh.vertices = np.column_stack([e[:, 0], s * v[:, 1] + up, -e[:, 1]])

# 對齊舊模型：舊表面是新表面的一部分，所以拿舊的點對新的（只轉和移），再用反變換把新模型放回舊座標。
# 對不上（重疊太少或要移太遠）就照 GPS 的位置，不硬套。
Tanchor = None
if anchor:
    from scipy.spatial import cKDTree
    old = trimesh.load(anchor, force='mesh').sample(20000); new = mesh.sample(20000)
    T, _, _ = trimesh.registration.icp(old, new, max_iterations=100, scale=False, reflection=False)
    d = cKDTree(new).query(trimesh.transform_points(old, T))[0]
    ok = (d < 0.3).mean() > 0.6 and np.linalg.norm(T[:3, 3]) < 6
    report['anchor'] = {'overlap': round(float((d < 0.3).mean()), 2), 'shiftM': round(float(np.linalg.norm(T[:3, 3])), 2), 'used': bool(ok)}
    if ok: mesh.apply_transform(np.linalg.inv(T)); Tanchor = T
report['center'] = mesh.bounds.mean(0).round(1).tolist()

# 拍照的位置與朝向（岩場座標）：檢視時從站在岩壁下的人看過去的角度打開
def place(v):
    e = s * (Rm @ np.array([v[0], -v[2]])) + t
    q = np.array([e[0], s * v[1] + up, -e[1]])
    return (np.linalg.inv(Tanchor) @ np.append(q, 1))[:3] if Tanchor is not None else q


views = []
for p in poses:
    c = np.array(p['position'])
    d = place(c + p['fwd']) - place(c)
    # 相機的右、上、後三個軸（岩場座標）：投影路線用；place 帶比例，除掉長度只留方向
    x, y, z, w = p['quaternion']
    Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    axes = [place(c + Rg @ Rq[:, k]) - place(c) for k in range(3)]
    rot = np.column_stack([a / np.linalg.norm(a) for a in axes])
    views.append({'name': Path(p['name']).stem, 'at': place(c).round(3).tolist(), 'dir': (d / np.linalg.norm(d)).round(3).tolist(), 'rot': rot.round(5).tolist()})
raw = out.with_suffix('.raw.glb'); mesh.export(raw)
subprocess.run([os.environ.get('GLTFPACK', 'gltfpack'), '-i', str(raw), '-o', str(out), '-cc', '-kn'], check=True, capture_output=True)
report['facesOut'] = len(mesh.faces); report['glbBytes'] = out.stat().st_size
out.with_suffix('.json').write_text(json.dumps({**report, 'origin': list(ORIGIN), 'views': views, 'rotation2d': Rm.tolist(), 'translation': t.tolist()}, ensure_ascii=False, indent=1))
print(json.dumps(report, ensure_ascii=False))
