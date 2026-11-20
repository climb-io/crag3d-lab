"""戶外 3D Topo 實驗：照片上畫好的路線（紅色虛線）投到 3D 模型表面。全自動，對不上就丟並記原因。

1. 描紅線：照片裡很鮮的紅色、垂直方向接起虛線，每一列取紅點的中位數，由下往上成一條線
2. 找出這張照片在 3D 裡的位置：跟模型用到的照片（位置已知）做 SIFT 配對，配到的點沿那張照片的視線打到模型表面
   得到 3D 點，再用 PnP 求這張照片的位置（焦距不知道，試一串取內點最多的）
3. 從求出的位置把紅線每個點的視線打到模型表面，得到 3D 路線

用法：route3d.py <工作資料夾> <路線 json> <輸出資料夾>
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, pycolmap, trimesh

D, routes_file, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 1200 / 1600   # 模型照片縮到長邊 1200 做配對


sys.path.insert(0, str(Path(__file__).parent)); from redline import trace, wall_run   # 同一資料夾的共用模組


# 模型、相機（位置與轉向來自 clean.py 的 views，內參來自 COLMAP）
areas = json.loads((D / 'groups/areas.json').read_text())
meshes, cams = {}, []
for a in areas:
    m = trimesh.util.concatenate([trimesh.load(D / 'groups' / g / 'wall.raw.glb', force='mesh') for g in a['groups']])
    meshes[a['area']] = (m, trimesh.ray.ray_triangle.RayMeshIntersector(m))
    for g in a['groups']:
        rec = pycolmap.Reconstruction(str(D / 'work/sfm' / g[1:]))
        intr = {im.name[:-4]: rec.cameras[im.camera_id].params[:3] for im in rec.images.values()}
        for v in json.loads((D / 'groups' / g / 'wall.json').read_text())['views']:
            if v['name'] in intr: cams.append({**v, 'area': a['area'], 'f': intr[v['name']][0], 'c': intr[v['name']][1:3]})


def hit(area, origins, dirs):
    """每條視線打到模型的第一個點；沒打到的是 nan"""
    m, ri = meshes[area]
    out = np.full((len(dirs), 3), np.nan)
    loc, idx, _ = ri.intersects_location(np.broadcast_to(origins, dirs.shape), dirs, multiple_hits=False)
    out[idx] = loc
    return out


def cam_rays(c, uv):   # 模型照片：像素 → 岩場座標的視線（相機右、上、後）
    d = np.column_stack([(uv[:, 0] - c['c'][0]) / c['f'], -(uv[:, 1] - c['c'][1]) / c['f'], -np.ones(len(uv))])
    d = d @ np.array(c['rot']).T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def project(c, X):   # 岩場座標 → 模型照片的像素
    p = (X - c['at']) @ np.array(c['rot'])
    return np.column_stack([c['c'][0] + c['f'] * p[:, 0] / -p[:, 2], c['c'][1] - c['f'] * p[:, 1] / -p[:, 2]]), -p[:, 2]


sift = cv2.SIFT_create(nfeatures=4000)
for c in cams:
    im = cv2.imread(str(D / 'work/img' / f"{c['name']}.jpg"), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    c['kp'], c['des'] = sift.detectAndCompute(im, None)
matcher = cv2.BFMatcher(cv2.NORM_L2)
routes = json.loads(routes_file.read_text())
results = []
for r in routes:
    res = {k: r[k] for k in ('id', 'number', 'name', 'grade', 'color')}
    bgr = cv2.imread(str(routes_file.parent / r['file']))
    line = trace(bgr)
    if line is None: results.append({**res, 'drop': '描不到紅線'}); continue
    kp, des = sift.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)
    # 配對：每張模型照片的 RANSAC 內點數，取最多的 3 張（同一區）
    scored = []
    for c in cams:
        ms = [m for m, n in matcher.knnMatch(des, c['des'], k=2) if m.distance < 0.75 * n.distance]
        if len(ms) < 15: continue
        a = np.float32([kp[m.queryIdx].pt for m in ms]); b = np.float32([c['kp'][m.trainIdx].pt for m in ms]) / SCALE
        F, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 3, 0.999)
        if F is None: continue
        mask = mask.ravel().astype(bool)
        scored.append((int(mask.sum()), c, a[mask], b[mask]))
    scored.sort(key=lambda x: -x[0])
    if not scored or scored[0][0] < 30: results.append({**res, 'drop': '模型裡沒有這塊岩壁（配對太少）', 'bestMatches': scored[0][0] if scored else 0}); continue
    area = scored[0][1]['area']
    top = [s for s in scored if s[1]['area'] == area][:3]
    P2, P3 = [], []
    for _, c, a, b in top:
        X = hit(area, np.array(c['at']), cam_rays(c, b))
        ok = ~np.isnan(X[:, 0]); P2.append(a[ok]); P3.append(X[ok])
    P2, P3 = np.concatenate(P2), np.concatenate(P3)
    if len(P2) < 12: results.append({**res, 'drop': '配到的點打不到模型表面'}); continue
    # PnP：焦距不知道，試一串
    h, w = bgr.shape[:2]; best = None
    for fk in np.linspace(0.7, 1.8, 23):
        K = np.array([[fk * max(w, h), 0, w / 2], [0, fk * max(w, h), h / 2], [0, 0, 1]])
        ok, rv, tv, inl = cv2.solvePnPRansac(P3, P2, K, None, reprojectionError=6, iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
        if ok and inl is not None and (best is None or len(inl) > best[0]): best = (len(inl), K, rv, tv, fk)
    if best is None or best[0] < 12: results.append({**res, 'drop': '求不出這張照片的拍攝位置', 'pnpInliers': best[0] if best else 0}); continue
    n_in, K, rv, tv, fk = best
    Rcv = cv2.Rodrigues(rv)[0]; C = (-Rcv.T @ tv).ravel()
    d = (np.column_stack([line, np.ones(len(line))]) @ np.linalg.inv(K).T) @ Rcv
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    X, ratio = wall_run(hit(area, C, d))
    if ratio < 0.6: results.append({**res, 'drop': f'紅線連續打到模型的只有 {ratio:.0%}', 'area': area}); continue
    res.update(area=area, points=X.round(3).tolist(), hitRatio=round(ratio, 2), pnpInliers=int(n_in), matched=[s[1]['name'] for s in top], focal=round(float(fk), 2),
               camera=C.round(2).tolist(), pose={'K': K.round(2).tolist(), 'rvec': rv.ravel().tolist(), 'tvec': tv.ravel().tolist()})
    results.append(res)
    # 核對圖：左邊舊照片的紅線，右邊把 3D 路線投回配對最多的那張模型照片（綠線）
    c = top[0][1]; uv, z = project(c, X); uv = uv[z > 0]
    cur = cv2.imread(str(D / 'work/img' / f"{c['name']}.jpg"))
    cv2.polylines(cur, [uv.astype(np.int32)], False, (0, 255, 0), 6)
    hh = 800; left = cv2.resize(bgr, (int(w * hh / h), hh)); right = cv2.resize(cur, (int(cur.shape[1] * hh / cur.shape[0]), hh))
    cv2.imwrite(str(OUT / f"check-{r['number']}-{r['id']}.jpg"), np.hstack([left, right]))
    print(r['symbol'], '區', area, '內點', n_in, '打到', f"{ratio:.0%}", flush=True)
(OUT / 'routes3d.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
from collections import Counter
print(Counter(x.get('drop', '投上了') for x in results))
