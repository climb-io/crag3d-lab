"""戶外 3D Topo：建模真照片的相機（sfm_to_model.py）用紋理再對準一次。

ICP 只讓點貼在岩面上；岩壁接近平面，整片沿著岩面左右、上下滑也一樣貼，所以相機會沿著岩壁偏 0.3 m 左右
（在真照片的相機位置拍模型快照，岩縫、黑色條紋跟真照片錯開約 8%）。這裡用紋理找出這個滑動：
1. 挑 N 張真照片（沿著岩壁平均分），在它們的相機位置拍模型快照（snap.html 的 views=）。
2. 真照片跟快照用 LoFTR 配對（同一個位置拍的，配得到上千點）；同一個像素各自順著視線讀模型的深度，
   真照片那一點（現在的相機看到的位置）↔ 快照那一點（模型上真正的位置）就是一組對應。
3. 全部相機的對應一起解相似變換（Umeyama，RANSAC 去掉錯配），套到每個相機，輸出新的 cameras.json。
   大砲岩實測：每台相機偏的量不一樣（大多往上 0.25–0.7 m），一個整體變換解不掉，所以改用 refine：
4. refine：每張真照片各自用快照上的 3D 位置（快照相機＋模型深度）配真照片的像素，PnP（真照片自己的焦距）重算相機。

用法：
  bridge_fix.py views <cameras.json> <輸出 views.json> [N | 名稱清單檔]
  bridge_fix.py fit <cameras.json> <views.json> <快照資料夾> <真照片資料夾> <模型.spz> <輸出 cameras.json>
  bridge_fix.py refine <cameras.json> <views.json> <快照資料夾> <真照片資料夾> <模型.spz> <輸出 cameras.json>
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read
from photo_pose import matcher, fit, depth_map, DS, MATCH_W, SOLID

N, HALF, NEAR = 40, 2, 1.0


def K_of(c, k=1):
    return np.array([[c['f'] / k, 0, c['cx'] / k], [0, c['f'] / k, c['cy'] / k], [0, 0, 1]])


def views(cam_file, out, n=N):
    cams = sorted(json.loads(Path(cam_file).read_text())['cameras'], key=lambda c: c['C'][0])
    if Path(str(n)).exists():
        want = set(Path(n).read_text().split()); pick = [c for c in cams if c['name'] in want]
    else: pick = [cams[int(i)] for i in np.linspace(0, len(cams) - 1, int(n))]
    vs = [{'position': np.round(c['C'], 3).tolist(), 'target': np.round(np.array(c['C']) + np.array(c['R'][2]) * 5, 3).tolist(),
           'fov': float(np.degrees(2 * np.arctan(c['h'] / 2 / c['f']))), 'w': c['w'] // HALF, 'h': c['h'] // HALF, 'file': f"brg-{c['name']}.jpg", 'name': c['name']} for c in pick]
    Path(out).write_text(json.dumps(vs))
    print(f'{len(vs)} 張 → {out}')


def look(C, T):
    f = (T - C) / np.linalg.norm(T - C); r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    return np.stack([r, -u, f])


def similarity(a, b):
    ma, mb = a.mean(0), b.mean(0); A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (A ** 2).sum() * len(a)
    return s, R, mb - s * R @ ma


def fit_all(cam_file, views_file, snap_dir, img_dir, spz, out):
    cj = json.loads(Path(cam_file).read_text()); cams = {c['name']: c for c in cj['cameras']}
    sp = read(spz); Q = sp['xyz'][sp['alpha'] >= SOLID * 255]
    run = matcher(); A, B = [], []
    for v in json.loads(Path(views_file).read_text()):
        c = cams[v['name']]; f = Path(snap_dir) / v['file']
        if not f.exists(): continue
        photo, snap = cv2.imread(str(Path(img_dir) / f"{v['name']}.jpg")), cv2.imread(str(f))
        a, sa, ta = fit(photo, MATCH_W); b, sb, tb = fit(snap, MATCH_W)
        k0, k1, conf = run(a, b); ok = conf > 0.5
        k0, k1 = k0[ok] / [sa, ta], k1[ok] / [sb, tb]
        if len(k0) < 30: continue
        _, m = cv2.findFundamentalMat(k0, k1, cv2.USAC_MAGSAC, 2.0, 0.999)
        if m is None: continue
        k0, k1 = k0[m.ravel() > 0], k1[m.ravel() > 0]
        # 真照片：現在的相機（含旋轉、主點）；快照：snap.html 的 lookAt（沒有側傾）
        R0, C = np.array(c['R']), np.array(c['C']); K0 = K_of(c)
        R1 = look(C, np.array(v['target'])); K1 = K_of({'f': c['f'] / HALF, 'cx': v['w'] / 2, 'cy': v['h'] / 2})
        got = []
        for R, K, kp, w, h in ((R0, K0, k0, c['w'], c['h']), (R1, K1, k1, v['w'], v['h'])):
            D = depth_map(Q, R, C, np.diag([1 / DS, 1 / DS, 1]) @ K, int(w / DS), int(h / DS), NEAR, clamp=False)
            z = D[np.clip((kp[:, 1] / DS).astype(int), 0, D.shape[0] - 1), np.clip((kp[:, 0] / DS).astype(int), 0, D.shape[1] - 1)]
            got.append(C + ((np.column_stack([kp, np.ones(len(kp))]) @ np.linalg.inv(K).T) * z[:, None]) @ R)
        good = np.isfinite(got[0][:, 0]) & np.isfinite(got[1][:, 0])
        A.append(got[0][good]); B.append(got[1][good])
        print(v['name'], '配對', int(good.sum()), '平均位移', np.round(np.median(got[1][good] - got[0][good], 0), 3).tolist(), flush=True)
    A, B = np.concatenate(A), np.concatenate(B)
    rng = np.random.default_rng(0); best = None
    for _ in range(300):   # RANSAC：錯配、深度跳到別的面的點去掉
        i = rng.choice(len(A), 50, replace=False); s, R, t = similarity(A[i], B[i])
        inl = np.linalg.norm(s * A @ R.T + t - B, axis=1) < 0.08
        if best is None or inl.sum() > best.sum(): best = inl
    s, R, t = similarity(A[best], B[best])
    err0, err1 = np.median(np.linalg.norm(A - B, axis=1)), np.median(np.linalg.norm(s * A @ R.T + t - B, axis=1))
    print(f'對應 {len(A)} 組，內點 {best.sum()}；修正：縮放 {s:.4f}、旋轉 {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.2f}°、平移 {np.round(t, 3).tolist()}；誤差中位數 {err0:.3f} → {err1:.3f} m')
    for c in cj['cameras']:
        c['C'] = (s * R @ np.array(c['C']) + t).round(4).tolist(); c['R'] = (np.array(c['R']) @ R.T).round(6).tolist()
    cj['scale'] *= s
    Path(out).write_text(json.dumps(cj))
    print(f'→ {out}')


def refine(cam_file, views_file, snap_dir, img_dir, spz, out):
    cj = json.loads(Path(cam_file).read_text()); cams = {c['name']: c for c in cj['cameras']}
    sp = read(spz); Q = sp['xyz'][sp['alpha'] >= SOLID * 255]
    run = matcher(); moved = []
    for v in json.loads(Path(views_file).read_text()):
        c = cams[v['name']]; f = Path(snap_dir) / v['file']
        if not f.exists(): continue
        photo, snap = cv2.imread(str(Path(img_dir) / f"{v['name']}.jpg")), cv2.imread(str(f))
        a, sa, ta = fit(photo, MATCH_W); b, sb, tb = fit(snap, MATCH_W)
        k0, k1, conf = run(a, b); ok = conf > 0.5
        k0, k1 = k0[ok] / [sa, ta], k1[ok] / [sb, tb]
        C = np.array(c['C']); R1 = look(C, np.array(v['target'])); K1 = K_of({'f': c['f'] / HALF, 'cx': v['w'] / 2, 'cy': v['h'] / 2})
        D = depth_map(Q, R1, C, np.diag([1 / DS, 1 / DS, 1]) @ K1, int(v['w'] / DS), int(v['h'] / DS), NEAR, clamp=False)
        z = D[np.clip((k1[:, 1] / DS).astype(int), 0, D.shape[0] - 1), np.clip((k1[:, 0] / DS).astype(int), 0, D.shape[1] - 1)]
        good = np.isfinite(z)
        if good.sum() < 30: print(v['name'], '配對太少', int(good.sum())); continue
        X = C + ((np.column_stack([k1[good], np.ones(good.sum())]) @ np.linalg.inv(K1).T) * z[good, None]) @ R1
        K0 = K_of(c); R0 = np.array(c['R'])
        rvec0, tvec0 = cv2.Rodrigues(R0)[0], (-R0 @ C).reshape(3, 1)
        okp, rvec, tvec, inl = cv2.solvePnPRansac(X, k0[good], K0, None, rvec0, tvec0, useExtrinsicGuess=True, reprojectionError=6, iterationsCount=2000)
        if not okp or inl is None or len(inl) < 30: print(v['name'], 'PnP 失敗'); continue
        R = cv2.Rodrigues(rvec)[0]; Cn = (-R.T @ tvec).ravel()
        moved.append(np.linalg.norm(Cn - C))
        print(v['name'], '內點', len(inl), '/', int(good.sum()), '移動', np.round(Cn - C, 3).tolist(), flush=True)
        c['C'], c['R'] = Cn.round(4).tolist(), R.round(6).tolist()
    Path(out).write_text(json.dumps(cj))
    print(f'{len(moved)} 張重算，移動中位數 {np.median(moved):.3f} m → {out}')


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    {'views': views, 'fit': fit_all, 'refine': refine}[cmd](*a)
