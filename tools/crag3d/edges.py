"""戶外 3D Topo 實驗：修岩壁的上緣和底部，切出平順的邊。

模型的三角形約 25 cm，只能整個刪或留的話邊緣一定是鋸齒。改成每個頂點記「留」值（0～1，0.5 是切線），
存在 GLB 頂點顏色（COLOR_0）的 alpha；檢視頁畫的時候在三角形內沿 0.5 切開，邊緣是平順的線。整個在切線外的面直接刪，檔案不變大。

- 底部：拍照的人站在地上，相機離地約 1.4 m；每個頂點取水平最近 8 台岩壁下的相機（站點分類見 filter.py）裡最低的那台推出地面高度，留值照高度線性變化，
  切線在地面下 BELOW（1 m，岩壁腳下的石塊留著，路線起點的牌子才不會浮在空中），所以底邊是一條沿著地面的線。
- 上緣：岩頂只能從岩壁下斜著拍到，那裡沒有照片正面看過，貼圖被拉長、混進天空和樹叢。每個面算照片看它最清楚的程度
  ＝ max(焦距 × cos(視線與法向的夾角) ÷ 距離)（一公尺在照片上幾個像素），跟鄰近的面平均 3 圈；低於中位數 SHARP 倍的切掉。
- 只切跟外緣連在一起的整片；牆中間沒看清楚的多半是被擋住（這裡沒算遮擋），不挖洞。
- 切完跟主體分開的小片（不到全部的 PIECE）整片刪掉：多半是岩頂懸空的薄片。

- 拼接：給了另一塊模型（例如岩壁上的照片另外建的上半部，拼在岩壁下那塊上面）時，只補另一塊上方的空缺：
  離地不到 LIFT（地面用另一塊的相機推）、跟它的表面距離不到 GAP、或比水平 REACH 內它的最高點還低的部分切掉
  （從岩壁上往下拍會建出岩壁前的地面植物）；這段一樣用留值切，接縫是平順的線。

- 路線底下不切：給了路線（--routes，crag.json 或 routes3d.json）時，離路線 KEEP_R 公尺內的岩面一律留著（上緣往內收、
  看不清楚都不切），路線不會畫到被切掉的地方（像畫在天空裡）。

- 只補縫（--gaps）：另一塊也是從岩壁下拍的（不會有岩壁前的植物薄片）時，只切跟它重疊（GAP 內）的部分，其他都留著補它的洞。

用法：edges.py [--routes 路線檔] [--gaps] <工作資料夾> <群 id，例如 g0> [SHARP，預設 0.35] [優先的模型 wall.edge.raw.glb，幾塊用逗號分開]
輸出：groups/<群>/wall.edge.raw.glb、wall.edge.glb
"""
import io, json, subprocess, sys
from pathlib import Path
import numpy as np, pycolmap, trimesh
from PIL import Image
from scipy.spatial import cKDTree

args = sys.argv[1:]
routes_file, gaps_only = None, False
while args[:1] in (['--routes'], ['--gaps']):
    if args[0] == '--routes': routes_file, args = Path(args[1]), args[2:]
    else: gaps_only, args = True, args[1:]
sys.argv = [sys.argv[0], *args]
D, g = Path(sys.argv[1]), sys.argv[2]
SHARP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
others = sys.argv[4].split(',') if len(sys.argv) > 4 else []
other = trimesh.util.concatenate([trimesh.load(o, force='mesh', process=False) for o in others]) if others else None
EYE, BELOW, SOFT_G, SOFT_C, INSET, SOFT_B, PIECE, GAP, SOFT_O, REACH, LIFT, KEEP_R = 1.4, 1.0, 0.3, 0.15, 0.6, 0.3, 0.03, 0.3, 0.2, 1.0, 3.0, 0.8
G = D / 'groups' / g
mesh = trimesh.load(G / 'wall.raw.glb', force='mesh', process=False)
W = D / 'work' if (D / 'work').exists() else D   # 舊版本的備份沒有 work/ 這一層
rec = pycolmap.Reconstruction(str(W / 'sfm' / g[1:]))
cam = {im.name[:-4]: rec.cameras[im.camera_id] for im in rec.images.values()}
views = [v for v in json.loads((G / 'wall.json').read_text())['views'] if v['name'] in cam]
V, F = mesh.vertices, mesh.faces
_, vid = np.unique(np.round(V, 6), axis=0, return_inverse=True); vid = vid.reshape(-1)   # 貼圖接縫的頂點分兩份：位置相同算同一點
fm = vid[F]; nv = vid.max() + 1
adj = trimesh.graph.face_adjacency(faces=fm)

# 上緣：每個面最清楚被看到的程度（像素／公尺），跟鄰近的面平均 3 圈，再平均到頂點
C, N = mesh.triangles_center, mesh.face_normals
best = np.zeros(len(F))
for v in views:
    k = cam[v['name']]; f, cx, cy = k.params[:3]
    p = (C - v['at']) @ np.array(v['rot']); z = -p[:, 2]
    u, w = cx + f * p[:, 0] / np.maximum(z, 1e-6), cy - f * p[:, 1] / np.maximum(z, 1e-6)
    to = np.array(v['at']) - C; dist = np.linalg.norm(to, axis=1)
    ok = (z > 0.3) & (u >= 0) & (u < k.width) & (w >= 0) & (w < k.height)
    best = np.maximum(best, np.where(ok, f * np.abs((N * to).sum(1)) / dist ** 2, 0))
sm = best.copy()
for _ in range(3):
    acc, cnt = sm.copy(), np.ones(len(sm))
    np.add.at(acc, adj[:, 0], sm[adj[:, 1]]); np.add.at(acc, adj[:, 1], sm[adj[:, 0]])
    np.add.at(cnt, adj[:, 0], 1); np.add.at(cnt, adj[:, 1], 1); sm = acc / cnt
s_sum, s_cnt = np.zeros(nv), np.zeros(nv)
np.add.at(s_sum, fm.ravel(), np.repeat(sm, 3)); np.add.at(s_cnt, fm.ravel(), 1)
keep_c = np.clip((s_sum / s_cnt / np.median(sm) - SHARP) / SOFT_C + 0.5, 0, 1)

# 底部：地面高度
P = np.zeros((nv, 3)); P[vid] = V
top = {r['name'] for r in json.loads((W / 'filter.json').read_text()) if r.get('top')}
at = np.array([v['at'] for v in views if v['name'] not in top])   # 地面只看岩壁下的站：岩壁上的站比地面高很多
if len(at):
    _, idx = cKDTree(at[:, [0, 2]]).query(P[:, [0, 2]], k=min(8, len(at)))
    ground = at[idx, 1].min(axis=1) - EYE - BELOW   # 取最低的那台：有些照片站在較高的平台上拍
else:   # 只有岩壁上的照片（岩壁上半部那塊）：沒有地面可切
    ground = np.full(nv, -np.inf)
keep_g = np.clip((P[:, 1] - ground) / SOFT_G + 0.5, 0, 1)
# 上緣與兩側的外緣：建模的外緣本身就是鋸齒（三角形大）、岩頂常有天空和樹叢的薄片；從外緣往內收 INSET，
# 收進去的那條線比原本的齒平順，比 2 倍 INSET 窄的薄片也一起去掉。底部的外緣不算（底部照地面切）
e0 = np.sort(np.concatenate([fm[:, [0, 1]], fm[:, [1, 2]], fm[:, [2, 0]]]), axis=1)
ue, ec = np.unique(e0, axis=0, return_counts=True)
rim = np.unique(ue[ec == 1]); rim = rim[P[rim, 1] > ground[rim] + 1.0]
d_rim = cKDTree(P[rim]).query(P)[0] if len(rim) else np.full(nv, 1e9)
keep_b = np.clip((d_rim - INSET) / SOFT_B + 0.5, 0, 1)
keep_v = np.minimum(np.minimum(keep_c, keep_g), keep_b)
if routes_file:
    rs = json.loads(routes_file.read_text()); rs = rs['routes'] if isinstance(rs, dict) else rs
    rp = np.array([q for r in rs for q in r.get('points') or []])
    if len(rp): keep_v[cKDTree(rp).query(P)[0] < KEEP_R] = 1

# 只切跟外緣連在一起的整片
cut = (keep_v[fm] < 0.5).any(1)
e = np.sort(np.stack([fm[:, [0, 1]], fm[:, [1, 2]], fm[:, [2, 0]]], 1), axis=2)
key = e[..., 0].astype(np.int64) * nv + e[..., 1]
uk, kc = np.unique(key, return_counts=True)
on_edge = np.isin(key, uk[kc == 1]).any(1)
lab = trimesh.graph.connected_component_labels(adj[cut[adj[:, 0]] & cut[adj[:, 1]]], node_count=len(F))
inner = cut & ~np.isin(lab, np.unique(lab[cut & on_edge]))
keep_v[np.unique(fm[inner])] = 1
if other is not None:   # 跟另一塊重疊、或比它低的部分不管在哪都切
    S = trimesh.sample.sample_surface_even(other, 300000, seed=0)[0]
    d_o = cKDTree(S).query(P)[0]
    dd, ii = cKDTree(S[:, [0, 2]]).query(P[:, [0, 2]], k=64, distance_upper_bound=REACH)
    top_o = np.where(np.isfinite(dd), S[np.minimum(ii, len(S) - 1), 1], -np.inf).max(1)   # 水平 REACH 內另一塊的最高點
    at_o = np.array([v['at'] for v in json.loads((Path(others[0]).parent / 'wall.json').read_text())['views'] if v['name'] not in top])   # 用另一塊岩壁下的相機推地面
    _, near_o = cKDTree(at_o[:, [0, 2]]).query(P[:, [0, 2]], k=min(8, len(at_o)))
    floor = at_o[near_o, 1].min(axis=1) - EYE + LIFT
    keep_gap = np.clip((d_o - GAP) / SOFT_O + 0.5, 0, 1)
    keep_up = np.minimum(np.clip((P[:, 1] - floor) / SOFT_O + 0.5, 0, 1),
                         np.maximum(keep_gap * (P[:, 1] > top_o), np.clip((P[:, 1] - top_o) / SOFT_O + 0.5, 0, 1)))
    keep_v = np.minimum(keep_v, keep_gap if gaps_only else keep_up)

# 整個在切線外的面刪掉；留值寫進頂點屬性
gone = (keep_v[fm] < 0.5).all(1)
lab = trimesh.graph.connected_component_labels(adj[~gone[adj[:, 0]] & ~gone[adj[:, 1]]], node_count=len(F))
size = np.bincount(lab[~gone], minlength=lab.max() + 1)
small = size[lab] < PIECE * (~gone).sum()
n_piece = int((~gone & small).sum())
gone |= small
n_blur, n_under = int(((keep_c[fm] < 0.5).all(1) & gone).sum()), int(((keep_g[fm] < 0.5).all(1) & gone).sum())
Fk = F[~gone]
used = np.unique(Fk); remap = np.full(len(V), -1); remap[used] = np.arange(len(used))
mat = mesh.visual.material
out_mesh = trimesh.Trimesh(V[used], remap[Fk], process=False)
tex = mat.baseColorTexture.convert('RGB'); buf = io.BytesIO(); tex.save(buf, 'JPEG', quality=80)   # 重新存檔時貼圖會變 PNG，改回 JPEG
out_mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv[used], material=trimesh.visual.material.PBRMaterial(
    baseColorTexture=Image.open(io.BytesIO(buf.getvalue())), metallicFactor=0, roughnessFactor=1))
k = keep_v[vid[used]].astype(np.float32); out_mesh.vertex_attributes['COLOR_0'] = np.column_stack([np.ones((len(k), 3), np.float32), k])   # 留值放在頂點顏色的 alpha（gltfpack 會丟掉自訂屬性）
raw = G / 'wall.edge.raw.glb'
glb = out_mesh.export(file_type='glb').replace(b'"_COLOR_0"', b'"COLOR_0" ')   # trimesh 把自訂屬性名稱前面加底線；改回標準名稱，空白補齊長度
raw.write_bytes(glb)
out = G / 'wall.edge.glb'
subprocess.run([str(D / 'node/node_modules/.bin/gltfpack'), '-i', str(raw), '-o', str(out), '-cc', '-kn', '-kv'], check=True, capture_output=True)
print(f"{g}：{len(F)} 面 → {len(out_mesh.faces)}（上緣刪 {n_blur}、地面以下刪 {n_under}、分開的小片刪 {n_piece}），{out.stat().st_size / 1e3:.0f} KB")
