"""戶外 3D Topo：把一面岩壁的模型（一塊或幾塊）合成一個 GLB，只留看路線需要的「舞台」，讓人像站在現場看岩壁。

- 舞台：沿岩壁方向（路線點水平分布的主軸）路線範圍往外 SIDE 公尺、最高點往上 ABOVE、最低點往下 BELOW、
  岩面前方 FRONT（起攀的地面）、後方 BEHIND（岩頂往後延伸的部分）。框跟著岩壁轉。
  範圍外不硬切：FADE 公尺內頂點留值（COLOR_0 的 alpha）從 1 漸漸變 0，檢視時半透明淡出；全部是 0 的三角形刪掉。
  模型本身的大邊（周長超過 LOOP 公尺的破邊：地面、鄰牆沒拍到的地方）也往內 EDGE_FADE 公尺漸漸淡出，不留鋸齒狀的斷面；小洞不淡出（會變成一圈光暈）。
  原本就有留值的（edges.py 修過邊）取兩者較小的。trimesh 讀檔會丟掉 COLOR_0，這裡直接讀 GLB 的 accessor。
- 擋視線的植物：從起攀點前方站著看路線（幾個站位、眼睛高度），擋在路線前面、比路線所在的岩面前突出 OCC_FRONT 以上的
  綠色三角形，連同相連的綠色部分一起刪掉；連起來不到 OCC_MIN 平方公尺的不刪（長在岩面上的小草叢擋不住路線，刪了反而開洞）。岩石不刪：突出的岩柱、大石頭是現場真的會看到的。
- 輸出用 gltfpack 量化（不用 meshopt 壓縮，three.js 的 GLTFLoader 不必另外帶解碼器就能讀）。

用法：assemble.py [--clean] <crag.json（walls 要有 standAt，有 facing 就用）> <輸出.glb> <gltfpack> <模型 1 .raw.glb> [模型 2 …]
  --clean：不裁舞台、不淡出，改成直接刪掉岩壁前的東西（整片岩場拼接用，見 join.py）：擋視線的植物，
    加上比岩面突出 CLEAR_GROUND 以上、在起攀點高度附近（GROUND_H 內）的地面雜物、突出 CLEAR_GREEN 以上的植物（起攀點高度附近只刪隆起的、法線 y 不到 FLAT 的，平鋪在地面、岩台上的草不刪），
    綠色斑駁、逐面判斷會挖得坑坑洞洞：岩面前方的三角形有兩個鄰面要刪就一起刪（重複 SMOOTH 次），
    刪完把只剩一邊相連的鋸齒剝掉（重複 SMOOTH 次），最後刪掉跟主體不相連（共用邊才算相連）、不到 FRAG 平方公尺的碎塊。岩面的深度：路線點沿岩壁每 0.5 m 一格的中位數。突出的岩石不刪。
"""
import io, json, struct, subprocess, sys
from pathlib import Path
import cv2, numpy as np, trimesh
from PIL import Image

SIDE, ABOVE, BELOW, FRONT, BEHIND, FADE = 2.0, 1.5, 1.0, 4.0, 4.0, 1.5
OCC_FRONT, OCC_MIN, EYE, VIEW_PX, LOOP, EDGE_FADE = 1.0, 1.0, 1.6, 400, 4.0, 1.0
CLEAR_GROUND, GROUND_H, CLEAR_GREEN, FRAG, SMOOTH, FLAT = 3.0, 0.5, 1.0, 0.5, 3, 0.7   # 地面留 3 m：起攀的岩台、地面刪了岩壁會像浮在空中
clean = sys.argv[1] == '--clean'; args = sys.argv[1 + clean:]
crag_file, out, gltfpack, parts = Path(args[0]), Path(args[1]), args[2], args[3:]
crag = json.loads(crag_file.read_text())
pts = np.array([p for r in crag['routes'] for p in r.get('points') or []])
c0 = pts[:, [0, 2]].mean(0)
along = np.linalg.svd(pts[:, [0, 2]] - c0, full_matrices=False)[2][0]
wall = crag['walls'][0]
n = np.array(wall['facing'], float)[[0, 2]] if wall.get('facing') else np.array([-along[1], along[0]])
n /= np.linalg.norm(n)
if n @ (np.array(wall['standAt'])[[0, 2]] - c0) < 0: n = -n   # 朝站位那一邊是前方
along = np.array([n[1], -n[0]])
frame = lambda v: np.column_stack([(v[:, [0, 2]] - c0) @ along, v[:, 1], (v[:, [0, 2]] - c0) @ n])   # 沿岩壁、高度、往前
R = frame(pts)
lo = np.array([R[:, 0].min() - SIDE, R[:, 1].min() - BELOW, -BEHIND]); hi = np.array([R[:, 0].max() + SIDE, R[:, 1].max() + ABOVE, R[:, 2].max() + FRONT])


def keep_values(path):   # GLB 第一個網格的 COLOR_0 alpha
    b = Path(path).read_bytes()
    n_ = struct.unpack('<I', b[12:16])[0]; js = json.loads(b[20:20 + n_]); binc = b[20 + n_ + 8:]
    attr = js['meshes'][0]['primitives'][0]['attributes']
    if 'COLOR_0' not in attr: return None
    acc = js['accessors'][attr['COLOR_0']]; bv = js['bufferViews'][acc['bufferView']]
    start = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    return np.frombuffer(binc[start:start + acc['count'] * 16], np.float32).reshape(-1, 4)[:, 3].copy()


def green_faces(m, tex):
    uv = m.visual.uv[m.faces].mean(1); th, tw = tex.shape[:2]
    col = tex[np.clip(((1 - uv[:, 1]) * th).astype(int), 0, th - 1), np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1)]
    hsv = cv2.cvtColor(col[None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
    return (hsv[:, 0] >= 25) & (hsv[:, 0] <= 95) & (hsv[:, 1] >= 50)   # OpenCV 色相 0–180：黃綠到青綠


def face_pairs(V, F):
    """共用一條邊的三角形對；貼圖接縫的頂點分開存，用位置相同算同一點。"""
    _, inv = np.unique(np.round(V, 5), axis=0, return_inverse=True); f = inv.reshape(-1)[F]
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1); fi = np.tile(np.arange(len(F)), 3)
    o = np.lexsort((e[:, 1], e[:, 0])); e, fi = e[o], fi[o]
    same = (e[1:] == e[:-1]).all(1)
    return np.column_stack([fi[:-1][same], fi[1:][same]])


def count(pairs, mask, n):   # 每個三角形有幾個鄰面在 mask 裡
    return np.bincount(pairs[:, 0], mask[pairs[:, 1]], n) + np.bincount(pairs[:, 1], mask[pairs[:, 0]], n)


def clutter(m, green, pairs):
    """岩壁前的地面雜物和植物（--clean）。"""
    b = np.round(R[:, 0] / 0.5); xs = np.unique(b); ds = np.array([np.median(R[b == k, 2]) for k in xs])
    Fr = frame(m.triangles_center); front = Fr[:, 2] - np.interp(Fr[:, 0] / 0.5, xs, ds)
    low = Fr[:, 1] < R[:, 1].min() + GROUND_H   # 起攀點高度附近：地面、岩台上平鋪的草是自然的，只刪隆起的樹叢
    flat = m.face_normals[:, 1] > FLAT
    junk = (front > CLEAR_GROUND) & low | (front > CLEAR_GREEN) & green & ~(low & flat)
    for _ in range(SMOOTH): junk |= (front > CLEAR_GREEN) & (count(pairs, junk, len(junk)) >= 2)
    return junk


def tidy(keep, pairs, A):
    """剝掉只剩一邊相連的鋸齒，再刪不到 FRAG 平方公尺、不相連的碎塊；回傳要刪的。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    k = keep.copy()
    for _ in range(SMOOTH): k &= count(pairs, k, len(k)) >= 2
    p = pairs[k[pairs[:, 0]] & k[pairs[:, 1]]]
    lab = connected_components(coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(len(k), len(k))), directed=False)[1]
    k &= np.bincount(lab, np.where(k, A, 0))[lab] >= FRAG
    return keep & ~k


def occluders(m, green):
    """從起攀點前方站著看路線，擋在路線前面的綠色三角形（連同相連的綠色部分）。"""
    F, V = m.faces, m.vertices; Fr = frame(V[F].mean(1))
    wall_w = np.median(R[:, 2])
    front = Fr[:, 2] > wall_w + OCC_FRONT
    # 路線點加密到每 10 公分，當作要看到的目標
    tg = np.concatenate([np.linspace(a, b, max(int(np.linalg.norm(np.subtract(b, a)) / 0.1), 1), endpoint=False)
                         for r in crag['routes'] for a, b in zip(r['points'][:-1], r['points'][1:])])
    mid = tg.mean(0); eye_y = R[:, 1].min() + EYE
    seeds = np.zeros(len(F), bool)
    for u in np.linspace(R[:, 0].min(), R[:, 0].max(), 5):
        for d in (3.0, 6.0, 10.0):
            xz = c0 + u * along + (wall_w + d) * n
            P = np.array([xz[0], eye_y, xz[1]])
            fw = mid - P; fw /= np.linalg.norm(fw); rt = np.cross(fw, [0, 1, 0]); rt /= np.linalg.norm(rt); up = np.cross(rt, fw)
            foc = VIEW_PX / 2 / np.tan(np.radians(35))
            X = V - P; z = X @ fw; x = VIEW_PX / 2 + foc * (X @ rt) / np.where(z > 0.1, z, 1); y = VIEW_PX / 2 - foc * (X @ up) / np.where(z > 0.1, z, 1)
            zb = np.full((VIEW_PX, VIEW_PX), np.inf, np.float32); fid = np.full((VIEW_PX, VIEW_PX), -1)
            for i in np.nonzero((z[F] > 0.1).all(1) & (x[F].max(1) >= 0) & (x[F].min(1) < VIEW_PX) & (y[F].max(1) >= 0) & (y[F].min(1) < VIEW_PX))[0]:
                a, b, c = F[i]; xs, ys, zs = x[[a, b, c]], y[[a, b, c]], z[[a, b, c]]
                x0, x1 = int(max(np.floor(xs.min()), 0)), int(min(np.ceil(xs.max()), VIEW_PX - 1)); y0, y1 = int(max(np.floor(ys.min()), 0)), int(min(np.ceil(ys.max()), VIEW_PX - 1))
                dd = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
                if x0 > x1 or y0 > y1 or abs(dd) < 1e-12: continue
                gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
                l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / dd; l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / dd; l2 = 1 - l0 - l1
                zz = np.where((l0 >= 0) & (l1 >= 0) & (l2 >= 0), l0 * zs[0] + l1 * zs[1] + l2 * zs[2], np.inf)
                sub, sf = zb[y0:y1 + 1, x0:x1 + 1], fid[y0:y1 + 1, x0:x1 + 1]; nearer = zz < sub
                sub[nearer] = zz[nearer]; sf[nearer] = i
            Tz = (tg - P) @ fw; tx = VIEW_PX / 2 + foc * ((tg - P) @ rt) / Tz; ty = VIEW_PX / 2 - foc * ((tg - P) @ up) / Tz
            ok = (Tz > 0.1) & (tx >= 0) & (tx < VIEW_PX) & (ty >= 0) & (ty < VIEW_PX)
            xi, yi = tx[ok].astype(int), ty[ok].astype(int)
            hidden = Tz[ok] > zb[yi, xi] + 0.5
            hit = fid[yi[hidden], xi[hidden]]; hit = hit[hit >= 0]
            seeds[hit[green[hit] & front[hit]]] = True
    # 相連的綠色、突出的部分一起刪（整叢樹，不留破碎的一半）
    region = seeds.copy(); grow = green & front
    adj = m.face_adjacency
    for _ in range(200):
        a, b = adj[:, 0], adj[:, 1]
        add = np.zeros(len(F), bool); add[b[region[a] & grow[b]]] = True; add[a[region[b] & grow[a]]] = True
        add &= ~region
        if not add.any(): break
        region |= add
    pairs = adj[region[adj[:, 0]] & region[adj[:, 1]]]
    lab = trimesh.graph.connected_component_labels(pairs, node_count=len(F))
    area = np.bincount(lab, weights=np.where(region, m.area_faces, 0))
    return region & (area[lab] >= OCC_MIN)


def edge_fade(V, F):
    """每個頂點離大邊（周長超過 LOOP）多遠，換成 0–1 的留值；貼圖接縫的頂點在 OBJ 裡是分開的，用位置相同算同一點。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    _, inv = np.unique(np.round(V, 5), axis=0, return_inverse=True); inv = inv.reshape(-1); f = inv[F]; nv = inv.max() + 1
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1)
    u, cnt = np.unique(e, axis=0, return_counts=True); b = u[cnt == 1]   # 只屬於一個三角形的邊
    P = np.zeros((nv, 3)); P[inv] = V
    if not len(b): return np.ones(len(V), np.float32)
    nb, lab = connected_components(coo_matrix((np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(nv, nv)), directed=False)
    blen = np.linalg.norm(P[b[:, 0]] - P[b[:, 1]], axis=1); perim = np.bincount(lab[b[:, 0]], blen, nb)
    src = np.unique(b[perim[lab[b[:, 0]]] > LOOP])
    if not len(src): return np.ones(len(V), np.float32)
    w = np.linalg.norm(P[u[:, 0]] - P[u[:, 1]], axis=1)
    G = coo_matrix((w, (u[:, 0], u[:, 1])), shape=(nv, nv))
    d = dijkstra(G, directed=False, indices=src, min_only=True, limit=EDGE_FADE)
    t = np.clip(np.nan_to_num(d, posinf=EDGE_FADE) / EDGE_FADE, 0, 1)
    return (t * t * (3 - 2 * t))[inv].astype(np.float32)   # smoothstep


scene = trimesh.Scene()
for i, p in enumerate(parts):
    m = trimesh.load(p, force='mesh', process=False)
    tex = np.asarray(m.visual.material.baseColorTexture.convert('RGB'))
    green = green_faces(m, tex); occ = occluders(m, green)
    pairs = face_pairs(m.vertices, m.faces) if clean else None
    junk = clutter(m, green, pairs) & ~occ if clean else np.zeros(len(m.faces), bool)
    k = keep_values(p)
    if k is None: k = np.ones(len(m.vertices), np.float32)
    if not clean:
        v = frame(m.vertices)
        inside = np.minimum((v - lo).min(1), (hi - v).min(1))   # 到舞台邊的距離（外面是負的）
        k = np.minimum(k, np.clip(1 + inside / FADE, 0, 1)).astype(np.float32)
    keep = (k[m.faces] > 0).any(1) & ~occ & ~junk
    if clean: frag = tidy(keep, pairs, m.area_faces); keep &= ~frag
    if not clean: k = np.minimum(k, edge_fade(m.vertices, m.faces[keep]))   # 刪掉植物、舞台外之後剩下的大邊
    faces = m.faces[keep]
    used = np.unique(faces); remap = np.full(len(m.vertices), -1); remap[used] = np.arange(len(used))
    buf = io.BytesIO(); Image.fromarray(tex).save(buf, 'JPEG', quality=80)
    part = trimesh.Trimesh(m.vertices[used], remap[faces], process=False)
    part.visual = trimesh.visual.TextureVisuals(uv=m.visual.uv[used], material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.open(io.BytesIO(buf.getvalue())), metallicFactor=0, roughnessFactor=1))
    if not clean: part.vertex_attributes['COLOR_0'] = np.column_stack([np.ones((len(used), 3), np.float32), k[used]])
    scene.add_geometry(part, geom_name=f'part{i}')
    print(f'{p}：{len(m.faces)} 面 → {len(faces)}（擋視線的植物刪掉 {occ.sum()} 面、{m.area_faces[occ].sum():.1f} m²）' + (f'，岩壁前的雜物 {m.area_faces[junk].sum():.1f} m²、鋸齒和碎塊 {m.area_faces[frag].sum():.1f} m²' if clean else ''))
raw = out.with_suffix('.raw.glb')
raw.write_bytes(scene.export(file_type='glb').replace(b'"_COLOR_0"', b'"COLOR_0" '))
subprocess.run([gltfpack, '-i', str(raw), '-o', str(out), '-kn', '-kv'], check=True, capture_output=True)
print(f'{out}：{out.stat().st_size / 1e3:.0f} KB')
