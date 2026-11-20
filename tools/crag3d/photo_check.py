"""戶外 3D Topo：舊路線照片的拍攝位置對不對，用顏色檢查（photo_pose.py 的結果）。

岩壁上的黑色條紋、岩縫、植被在舊照片和模型上都看得到。從算出的拍攝位置拍一張模型快照（snap.html 的 views=），
兩張都拿掉紅線、縮到 W 寬、轉灰階取梯度（光線、季節不同時比亮度不準，比邊緣），算相關係數（NCC）：
位置對的話岩縫、條紋疊在一起，分數高；焦距、遠近錯了會錯開，分數低。

微調（refine）：舊照片跟快照用 LoFTR 配對（同一個位置拍的，錯開的只是剩下的誤差），快照上配到的點用模型深度換成 3D，
PnP（舊照片自己的焦距）重算拍攝位置，再投一次紅線；配到的點少於 MIN_REFINE 的不動。改完要重拍快照再看一次。

用法：
  photo_check.py refine <清單 todo-real.txt> <photo_pose 輸出資料夾> <快照資料夾> <views.json> <模型.spz>
  photo_check.py views <清單 todo-real.txt> <photo_pose 輸出資料夾> <輸出 views.json>   # 每張底圖一個鏡頭，交給 snap.html 拍 chk-<編號>.jpg
  photo_check.py score <清單 todo-real.txt> <photo_pose 輸出資料夾> <快照資料夾> [對照圖.jpg]
"""
import json, sys
from pathlib import Path
import cv2, numpy as np

W, MIN_REFINE = 300, 30


def shots(todo):
    seen, out = set(), []
    for line in Path(todo).read_text().split('\n'):
        if not line.strip(): continue
        tag, num, photo = line.split()
        if tag not in seen: seen.add(tag); out.append((num, photo))
    return out


def views(todo, res_dir, out):
    vs = []
    for num, photo in shots(todo):
        d = json.loads((Path(res_dir) / f'{num}.json').read_text())
        if 'R' not in d: continue
        h, w = cv2.imread(photo).shape[:2]; R, C = np.array(d['R']), np.array(d['position'])
        vs.append({'position': C.round(3).tolist(), 'target': (C + R[2] * 5).round(3).tolist(), 'fov': d['fov'], 'w': w, 'h': h, 'file': f'chk-{num}.jpg'})
    Path(out).write_text(json.dumps(vs))
    print(f'{len(vs)} 張 → {out}')


def edges(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] <= 8) | (hsv[..., 0] >= 172)) & (hsv[..., 1] > 120)).astype(np.uint8)
    img = cv2.inpaint(img, cv2.dilate(red, np.ones((11, 11))), 5, cv2.INPAINT_TELEA)
    g = cv2.cvtColor(cv2.resize(img, (W, int(W * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY).astype(float)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    return np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def score(todo, res_dir, snap_dir, sheet=None):
    rows = []
    for num, photo in shots(todo):
        f = Path(snap_dir) / f'chk-{num}.jpg'
        if not f.exists(): continue
        p, r = cv2.imread(photo), cv2.imread(str(f))
        s = ncc(edges(p), edges(cv2.resize(r, (p.shape[1], p.shape[0]))))
        print(num, f'{s:.3f}')
        if sheet: rows.append(np.hstack([cv2.resize(p, (240, 320)), cv2.resize(r, (240, 320))]))
    if sheet and rows: cv2.imwrite(sheet, np.vstack([np.hstack(rows[i:i + 4] + [np.zeros_like(rows[0])] * (4 - len(rows[i:i + 4]))) for i in range(0, len(rows), 4)]))


def refine(todo, res_dir, snap_dir, views_file, spz):
    sys.path.insert(0, str(Path(__file__).parent))
    from spz_crop import read
    from photo_pose import matcher, fit, cam, depth_map, project, overlay, K_of, MATCH_W, DS, SOLID
    sp = read(spz); Q = sp['xyz'][sp['alpha'] >= SOLID * 255]
    vs = {v['file']: v for v in json.loads(Path(views_file).read_text())}
    run = matcher(); lines = [l.split() for l in Path(todo).read_text().split('\n') if l.strip()]
    for num, photo in shots(todo):
        v = vs.get(f'chk-{num}.jpg'); f = Path(snap_dir) / f'chk-{num}.jpg'
        if not v or not f.exists(): continue
        p0 = cv2.imread(photo); r = cv2.imread(str(f)); h, w = p0.shape[:2]
        hsv = cv2.cvtColor(p0, cv2.COLOR_BGR2HSV); red = (((hsv[..., 0] <= 8) | (hsv[..., 0] >= 172)) & (hsv[..., 1] > 120)).astype(np.uint8)
        clean = cv2.inpaint(p0, cv2.dilate(red, np.ones((11, 11))), 5, cv2.INPAINT_TELEA)
        a, sa, ta = fit(clean, MATCH_W); b, sb, tb = fit(r, MATCH_W)
        k0, k1, c = run(a, b); k0, k1 = k0[c > 0.5] / [sa, ta], k1[c > 0.5] / [sb, tb]
        R1, C1, K1 = cam(v)
        D = depth_map(Q, R1, C1, np.diag([1 / DS, 1 / DS, 1]) @ K1, int(v['w'] / DS), int(v['h'] / DS), 1.0, clamp=False)
        z = D[np.clip((k1[:, 1] / DS).astype(int), 0, D.shape[0] - 1), np.clip((k1[:, 0] / DS).astype(int), 0, D.shape[1] - 1)]
        ok = np.isfinite(z)
        d = json.loads((Path(res_dir) / f'{num}.json').read_text()); K = K_of(d['fov'], w, h)
        if ok.sum() < MIN_REFINE: print(num, '配對太少，不動', int(ok.sum())); continue
        X = C1 + ((np.column_stack([k1[ok], np.ones(ok.sum())]) @ np.linalg.inv(K1).T) * z[ok, None]) @ R1
        R0 = np.array(d['R']); C0 = np.array(d['position'])
        good, rvec, tvec, inl = cv2.solvePnPRansac(X, k0[ok], K, None, cv2.Rodrigues(R0)[0], (-R0 @ C0).reshape(3, 1), useExtrinsicGuess=True, reprojectionError=6, iterationsCount=2000)
        if not good or inl is None or len(inl) < MIN_REFINE: print(num, 'PnP 失敗，不動'); continue
        R = cv2.Rodrigues(rvec)[0]; C = (-R.T @ tvec).ravel()
        print(num, '內點', len(inl), '/', int(ok.sum()), '移動', np.round(C - C0, 3).tolist(), flush=True)
        pose = {'R': R, 'C': C, 'K': K}
        for _, n2, ph in (l for l in lines if l[2] == photo):
            res = json.loads((Path(res_dir) / f'{n2}.json').read_text())
            route, share = project(Q, cv2.imread(ph), pose)
            if route is None: continue
            res.update(points=route.round(3).tolist(), share=round(share, 3), position=C.round(3).tolist(), R=R.round(6).tolist(), refined=len(inl))
            (Path(res_dir) / f'{n2}.json').write_text(json.dumps(res))
            cv2.imwrite(str(Path(res_dir) / f'{n2}.jpg'), overlay(cv2.imread(ph), pose, [(route, (40, 220, 40), 2)]))


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    {'views': views, 'score': score, 'refine': refine}[cmd](*a)
