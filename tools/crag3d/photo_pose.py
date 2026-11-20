"""戶外 3D Topo：舊路線照片（2D，紅色虛線）自己找出在 3D 模型上的位置（實驗）。

正面快照上人工描線會丟掉舊照片的拍攝角度，內角的線分不出在哪一面。這裡改成用舊照片自己的視角投線：
1. 參考圖：在那一區前方一圈位置（左右 YAWS 度、距離 DISTS 公尺、眼睛高度）對著岩壁拍參考圖（snap.html 的 views=）。
2. 比對：LoFTR（kornia，戶外權重，專門處理光線、季節不同的照片）把舊照片跟每張參考圖配對，RANSAC 留下的點最多的幾張是最像的角度。
3. 拍攝位置：參考圖上配到的點用它的深度（模型裡不透明的點畫 z-buffer）變成 3D 點，配上舊照片的像素，
   PnP 求舊照片的拍攝位置；焦距不知道，試一串、取內點最多的。
4. 投線：從求出的位置畫一張深度圖，紅線（redline.trace）每個像素讀出 3D 位置；打到後面的丟掉、沒打到的沿線內插（見 project）。
5. 對答案：跟參考的路線點（使用者確認過的）比，每個點到參考線的最近距離取平均。

用法：
  photo_pose.py views <crop.spz> <crag.json> <岩壁名> <輸出 views.json> [檔名前綴] [near]   # 產生參考圖的鏡頭清單（再交給 snap.html 拍）
  photo_pose.py batch <crop.spz> <參考 crag.json> <清單（區 路線編號 舊照片，照順序）> <參考圖資料夾> <輸出資料夾（含 views-<區>.json）>
    照路線順序一條一條對位：同一張底圖（只換紅線）的路線共用拍攝位置；先比上一張最像的參考圖附近，分數不夠（GOOD）才比整區。
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.ndimage import grey_erosion
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read
from redline import trace

YAWS, DISTS, EYE, FOV, W, H = range(-60, 61, 15), (8, 12, 16), 1.6, 60, 600, 800
NEAR_YAWS, NEAR_DISTS, NEAR_UP = range(-75, 76, 15), (4, 6, 9), 2.5
SOLID, FILL, MATCH_W, TOP, GOOD, SAME = 0.5, 5, 640, 3, 150, 0.2
FOV_FIXED = float(__import__('os').environ.get('FOV_FIXED', 0))   # 全部舊照片共用的視角（同一台相機）；0：每張自己試
DS, NEAR_Z, EDGE, RIM, JUMP, MED, OFF, OFF_K = 4, 1.5, 0.012, 0.05, 1.5, 15, 0.5, 0.1


def look(C, T):
    f = (T - C) / np.linalg.norm(T - C); r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    return np.stack([r, -u, f])


def K_of(fov, w, h):
    fy = h / 2 / np.tan(np.radians(fov) / 2)
    return np.array([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])


def cam(v):   # 參考圖的相機：模型畫的參考圖（position、target、fov）或建模真照片（R、C、焦距、主點，photo_real.py）
    if 'R' in v: return np.array(v['R']), np.array(v['C']), np.array([[v['f'], 0, v['cx']], [0, v['f'], v['cy']], [0, 0, 1]])
    C = np.array(v['position']); return look(C, np.array(v['target'])), C, K_of(v['fov'], v['w'], v['h'])


def depth_map(Q, R, C, K, w, h, near=0.3, clamp=True):
    X = (Q - C) @ R.T
    X = X[X[:, 2] > near]
    uv = X[:, :2] / X[:, 2:] @ K[:2, :2].T + K[:2, 2]
    i, j = np.round(uv[:, 1]).astype(int), np.round(uv[:, 0]).astype(int)
    ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    d = np.full(h * w, np.inf); np.minimum.at(d, i[ok] * w + j[ok], X[ok, 2])
    d = d.reshape(h, w)
    filled = grey_erosion(np.where(np.isinf(d), 1e9, d), size=(FILL, FILL))
    if not clamp: return np.where(np.isinf(d), np.where(filled < 1e9, filled, np.inf), d)   # 只補洞：參考圖的深度不能往相機偏（斜的岩面上會把點拉近，拍攝位置跟著被拉近、視角變寬）
    return np.where(np.isinf(d), np.where(filled < 1e9, filled, np.inf), np.minimum(d, filled + 0.15))


def views(spz, crag_file, wall, out, tag='ref', near=''):
    crag = json.loads(Path(crag_file).read_text())
    w = next(x for x in crag['walls'] if x['name'] == wall)
    pts = np.array([p for r in crag['routes'] if w['routes'][0] <= int(r['number']) <= w['routes'][1] for p in r.get('points') or []])
    T = pts.mean(0)
    f = np.array(w['facing'], float); f[1] = 0; f /= np.linalg.norm(f)
    ground = pts[:, 1].min()
    vs = []
    # near：舊照片常常貼近岩壁、從下往上拍，再補近距離、更斜、往上看的參考圖
    dists, yaws, ups = (NEAR_DISTS, NEAR_YAWS, (0, NEAR_UP)) if near else (DISTS, YAWS, (0,))
    for up in ups:
        for d in dists:
            for yaw in yaws:
                a = np.radians(yaw)
                dirv = np.array([f[0] * np.cos(a) + f[2] * np.sin(a), 0, -f[0] * np.sin(a) + f[2] * np.cos(a)])
                C = T + dirv * d; C[1] = ground + EYE; tgt = T + [0, up, 0]
                vs.append({'position': C.round(3).tolist(), 'target': tgt.round(3).tolist(), 'fov': FOV, 'w': W, 'h': H, 'file': f'{tag}-{len(vs):02d}.jpg'})
    Path(out).write_text(json.dumps(vs, ensure_ascii=False))
    print(f'{len(vs)} 張參考圖 → {out}')


def matcher():
    import torch, kornia.feature as KF
    m = KF.LoFTR(pretrained='outdoor').eval()
    def run(a, b):
        ta, tb = (torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)).float()[None, None] / 255 for x in (a, b))
        with torch.no_grad():
            o = m({'image0': ta, 'image1': tb})
        return o['keypoints0'].numpy(), o['keypoints1'].numpy(), o['confidence'].numpy()
    return run


def fit(img, w):   # 縮到寬 w（LoFTR 要 8 的倍數）
    h = int(round(img.shape[0] * w / img.shape[1] / 8)) * 8
    return cv2.resize(img, (w, h)), w / img.shape[1], h / img.shape[0]


def pose(Q, photo, refs, run):
    """舊照片 → 拍攝位置 (R, C, K) 與分數。refs：[(參考圖鏡頭, 參考圖)]"""
    ph, pw = photo.shape[:2]
    p_small, psx, psy = fit(photo, MATCH_W)
    scored = []
    for v, ref in refs:
        r_small, rsx, rsy = fit(ref, MATCH_W)
        k0, k1, conf = run(p_small, r_small)
        k0, k1 = k0[conf > 0.5], k1[conf > 0.5]
        if len(k0) < 8: scored.append((0, v, None, None)); continue
        _, mask = cv2.findFundamentalMat(k0, k1, cv2.USAC_MAGSAC, 2.0, 0.999)
        inl = mask.ravel().astype(bool) if mask is not None else np.zeros(len(k0), bool)
        scored.append((int(inl.sum()), v, k0[inl] / [psx, psy], k1[inl] / [rsx, rsy]))
    scored.sort(key=lambda x: -x[0])
    # 參考圖上配到的點 → 3D（用那張參考圖的深度），配上舊照片的像素
    obj, img = [], []
    for n, v, kp, kr in scored[:TOP]:
        if not n: continue
        R, C, K = cam(v)
        # 深度用 1/DS 的解析度算：近拍的真照片上模型的點很稀，原解析度的深度圖大半是洞
        w, h = int(v['w'] / DS), int(v['h'] / DS)
        D = depth_map(Q, R, C, np.diag([1 / DS, 1 / DS, 1]) @ K, w, h, clamp=False)
        z = D[np.clip((kr[:, 1] / DS).astype(int), 0, h - 1), np.clip((kr[:, 0] / DS).astype(int), 0, w - 1)]
        ok = np.isfinite(z)
        obj.append(C + ((np.column_stack([kr[ok], np.ones(ok.sum())]) @ np.linalg.inv(K).T) * z[ok, None]) @ R); img.append(kp[ok])
    if not obj: return None
    obj, img = np.concatenate(obj), np.concatenate(img)
    best, fovs = None, {}
    for fov in ([FOV_FIXED] if FOV_FIXED else np.arange(35, 86, 2.5)):   # 舊照片的焦距不知道：試一串視角（FOV_FIXED：全部照片共用一個）
        K = K_of(fov, pw, ph)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=8, iterationsCount=2000, flags=cv2.SOLVEPNP_EPNP)
        fovs[float(fov)] = len(inl) if ok and inl is not None else 0
        if ok and inl is not None and (best is None or len(inl) > best[0]): best = (len(inl), fov, rvec, tvec, K)
    if best is None: return None
    n_in, fov, rvec, tvec, K = best
    R = cv2.Rodrigues(rvec)[0]; C = (-R.T @ tvec).ravel()
    return {'R': R, 'C': C, 'K': K, 'fov': float(fov), 'fovs': fovs, 'inliers': n_in, 'matches': len(obj), 'best': scored[0][0], 'top': [v['file'] for _, v, *_ in scored[:TOP]]}


def project(Q, photo, p):
    """紅線從拍攝位置投到模型上 → 整條紅線的 3D 折線與真的打到模型的比例。
    - 紅線貼著岩角、畫在輪廓邊上時，視線擦過岩角打到後面或天空：深度取附近 EDGE 比例寬的範圍裡最近的面；
      還是比 RIM 比例寬的範圍裡最近的面遠 JUMP 公尺以上（或打到天空），就是畫在輪廓外一點點，改用那個面。
    - 舊照片站在岩壁前幾公尺拍：鏡頭前 NEAR_Z 公尺內的點（潑濺模型飄在空中的雜點）不算。
    - 沿著線的深度應該平順：離滑動中位數（MED 點）太遠的當成打到後面的岩壁，丟掉。
    - 沒打到的（模型破洞、草、輪廓外）用前後打到的深度沿線內插，頭尾用最近一點的深度延伸：起攀到地面、結束到岩頂都照舊照片畫的。"""
    ph, pw = photo.shape[:2]
    line = trace(photo)
    if line is None: return None, 0
    R, C, K = p['R'], p['C'], p['K']
    D0 = np.where(np.isinf(D0 := depth_map(Q, R, C, K, pw, ph, NEAR_Z)), 1e9, D0)
    yx = np.clip(np.round(line[:, 1]).astype(int), 0, ph - 1), np.clip(np.round(line[:, 0]).astype(int), 0, pw - 1)
    z, z_rim = (grey_erosion(D0, size=(r, r))[yx] for r in (max(3, int(EDGE * pw)) | 1, max(5, int(RIM * pw)) | 1))
    z = np.where(z > z_rim + JUMP, z_rim, z); z[z >= 1e9] = np.inf
    ok = np.isfinite(z)
    if ok.sum() < 2: return None, 0
    zs = np.where(ok, z, np.nan)
    med = np.array([np.nanmedian(zs[max(0, i - MED // 2):i + MED // 2 + 1]) for i in range(len(zs))])
    ok &= np.abs(zs - med) <= OFF + OFF_K * med
    if ok.sum() < 2: return None, 0
    i = np.arange(len(z)); z = np.interp(i, i[ok], z[ok])
    X = C + ((np.column_stack([line, np.ones(len(line))]) @ np.linalg.inv(K).T) * z[:, None]) @ R
    return X, float(ok.mean())


def overlay(photo, p, lines):
    vis = photo.copy()
    for pts, col, w in lines:
        uv = (np.asarray(pts) - p['C']) @ p['R'].T @ p['K'].T; uv = uv[uv[:, 2] > 0]
        cv2.polylines(vis, [(uv[:, :2] / uv[:, 2:]).astype(np.int32)], False, col, w)
    return vis


def same_shot(a, b):   # 同一張底圖（只換紅線）：拿掉紅色後縮小比灰階
    def sig(im):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        red = ((hsv[..., 0] <= 8) | (hsv[..., 0] >= 172)) & (hsv[..., 1] > 120)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float); g[red] = np.nan
        x = cv2.resize(np.nan_to_num(g, nan=np.nanmean(g)), (48, 64)); return (x - x.mean()) / x.std()
    return a.shape == b.shape and np.mean(np.abs(sig(a) - sig(b))) < SAME


def batch(spz, crag_file, todo_file, ref_dir, out_dir):
    """照路線順序一條一條對位：同一張底圖的路線共用拍攝位置；先比上一張底圖最像的參考圖附近，分數不夠才比整區"""
    s = read(spz); Q = s['xyz'][s['alpha'] >= SOLID * 255]
    truth = {r['number']: np.array(r['points']) for r in json.loads(Path(crag_file).read_text())['routes'] if r.get('points')}
    out = Path(out_dir); run = matcher()
    views = {}
    last = None   # (底圖, 拍攝位置, 區)
    for line_ in Path(todo_file).read_text().split('\n'):
        if not line_.strip(): continue
        wall, number, photo_file = line_.split()
        photo = cv2.imread(photo_file)
        if wall not in views: views[wall] = json.loads((out / f'views-{wall}.json').read_text())
        vs = views[wall]
        if last and last[2] == wall and same_shot(last[0], photo):
            p = last[1]; how = '同一張底圖'
        else:
            near = []
            if last and last[2] == wall and last[1] and 'R' not in vs[0]:
                i = int(last[1]['top'][0].split('-')[-1][:-4]); yaws = len(NEAR_YAWS if 'n' in last[1]['top'][0].split('-')[0] else YAWS)
                near = [k for k in range(len(vs)) if abs(k % yaws - i % yaws) <= 1]   # 上一張最像的那個方向左右各一格、各種距離
            refs = lambda ks: [(vs[k], cv2.imread(str(Path(ref_dir) / vs[k]['file']))) for k in ks]
            p = pose(Q, photo, refs(near), run) if near else None; how = '比附近'
            if not p or p['best'] < GOOD:
                p = pose(Q, photo, refs(range(len(vs))), run); how = '比整區'
        last = (photo, p, wall)
        res = {'number': number, 'how': how}
        if p:
            route, share = project(Q, photo, p)
            if route is not None and len(route):
                d = cKDTree(truth[number]).query(route)[0] if number in truth else np.array([np.nan])
                res.update(points=route.round(3).tolist(), share=round(share, 3), mean=round(float(d.mean()), 3), max=round(float(d.max()), 3))
                cv2.imwrite(str(out / f'{number}.jpg'), overlay(photo, p, [(truth.get(number, route), (255, 200, 0), 3), (route, (40, 220, 40), 2)]))
            res.update(inliers=p['inliers'], matches=p['matches'], best=p['best'], fov=p['fov'], fovs=p['fovs'], position=p['C'].round(3).tolist(), R=p['R'].round(6).tolist())
        (out / f'{number}.json').write_text(json.dumps(res))
        print(number, how, {k: res.get(k) for k in ('best', 'inliers', 'share', 'mean', 'max')}, flush=True)


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    views(*a) if cmd == 'views' else batch(*a)
