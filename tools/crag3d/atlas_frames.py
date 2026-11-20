"""戶外 3D Topo：挑好的照片（atlas64.json）做成 World Labs posedImages2Splats 的輸入（實驗）。

每張照片：
- 照片縮成 720×1280（原圖 900×1600，比例相同）。
- 拍攝位置：岩壁照片用建模的相機（cameras-train.json），岩頂照片用逐張對位的結果（top-register.json，register_photos.py）。
  相機慣例是 OpenCV（X_cam = R (X − C)）；World Labs 要 rub（右、上、後）的「相機到世界」旋轉，三欄是 R[0]、−R[1]、−R[2]，存成四元數 xyzw。
- 深度圖（公尺、float32 EXR，0 = 沒有表面）：模型的點（岩壁網格取的點 + 背景潑濺）用 z-buffer 投到照片上（不做光線測試）；
  點跟點之間的縫用點的大小決定蓋幾個像素（最多 MAX_R），近處還有縫的地方用 1/4 解析度的深度補，也擋住從縫裡漏出來的遠處點。
- 遮罩：atlas64.json 的 mask（照片名 → below 比例），那條線以下的深度留 0。
- 檢查圖：照片和深度疊在一起（對得準，岩壁邊緣會貼齊）。

用法：atlas_frames.py <atlas64.json> <相機檔[,相機檔…]> <top-register.json> <照片資料夾> <輸出資料夾> <模型點.spz> [背景.spz…]
輸出：<輸出>/img/*.jpg、depth/*.exr、check/*.jpg、frames.json（每張的位置、四元數、內參）
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import json, math, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read

W, H, F0, W0 = 720, 1280, 1103.066, 900
NEAR, FAR, MAX_R, COARSE, LEAK, MIN_INLIERS = 0.4, 80.0, 3, (4, 16), 1.25, 150


def quat(M):   # 旋轉矩陣 → xyzw（w ≥ 0）
    t = np.trace(M)
    if t > 0:
        s = math.sqrt(t + 1) * 2; q = [(M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(M))); j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + M[i, i] - M[j, j] - M[k, k]) * 2
        q = [0, 0, 0, (M[k, j] - M[j, k]) / s]; q[i] = s / 4; q[j] = (M[j, i] + M[i, j]) / s; q[k] = (M[k, i] + M[i, k]) / s
    q = np.array(q); q /= np.linalg.norm(q)
    return (-q if q[3] < 0 else q).round(6).tolist()


def zbuf(P, size, R, C, w, h, f, cx, cy):
    X = (P - C) @ R.T; z = X[:, 2]
    ok = (z > NEAR) & (z < FAR); X, z, size = X[ok], z[ok], size[ok]
    u = np.rint(X[:, 0] / z * f + cx).astype(np.int32); v = np.rint(X[:, 1] / z * f + cy).astype(np.int32)
    r = np.clip(np.ceil(0.7 * size * f / z) - 1, 0, MAX_R).astype(np.int32)
    ok = (u >= -MAX_R) & (u < w + MAX_R) & (v >= -MAX_R) & (v < h + MAX_R); u, v, z, r = u[ok], v[ok], z[ok].astype(np.float32), r[ok]
    d = np.full((h, w), np.inf, np.float32)
    for rr in range(MAX_R + 1):
        m = r >= rr
        for du in range(-rr, rr + 1):
            for dv in range(-rr, rr + 1):
                if max(abs(du), abs(dv)) != rr: continue
                uu, vv = u[m] + du, v[m] + dv; k = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                np.minimum.at(d, (vv[k], uu[k]), z[m][k])
    return d


def depth(P, size, R, C, f, cx, cy):
    d = zbuf(P, size, R, C, W, H, f, cx, cy)
    for k in COARSE:   # 先用 1/4、再用 1/16 解析度：縫補上、從縫漏出的遠點換成近處
        c = cv2.resize(zbuf(P, size, R, C, W // k, H // k, f / k, cx / k, cy / k), (W, H), interpolation=cv2.INTER_NEAREST)
        d = np.where(np.isfinite(d) & ~(d > c * LEAK), d, c)
    d[~np.isfinite(d)] = 0
    return d.astype(np.float32)


def main():
    pick, cams, top, img_dir, out, *spz = sys.argv[1:]
    pick = json.loads(Path(pick).read_text())
    cams = {c['name']: c for f in reversed(cams.split(',')) for j in [json.loads(Path(f).read_text())]
            for c in (j['cameras'] if isinstance(j, dict) else j) if 'name' in c and 'C' in c}   # 多個相機檔用逗號分隔，前面的優先
    top = json.loads(Path(top).read_text())
    names = pick['wall'] + [n for n, _ in pick['top']]
    P, S = [], []
    for f in spz:
        s = read(f); keep = s['alpha'] >= 40
        P.append(s['xyz'][keep].astype(np.float32)); S.append(np.exp(s['logs'][keep]).max(1).astype(np.float32))
    P, S = np.concatenate(P), np.concatenate(S)
    print(f'{len(P):,} 點', flush=True)
    out = Path(out)
    for d in ('img', 'depth', 'check'): (out / d).mkdir(parents=True, exist_ok=True)
    frames = []
    for n in names:
        c = top.get(n) if n not in cams else cams[n]
        if not c or 'C' not in c or c.get('inliers', MIN_INLIERS) < MIN_INLIERS: print(n, '沒有位置（或對位太弱），跳過'); continue
        R, C = np.array(c['R'], float), np.array(c['C'], float)
        img = cv2.imread(str(Path(img_dir) / f'{n}.jpg')); h0, w0 = img.shape[:2]
        f0, cx0, cy0 = c.get('f', F0 * w0 / W0), c.get('cx', w0 / 2), c.get('cy', h0 / 2)
        if c.get('w') and c['w'] != w0: k = w0 / c['w']; f0, cx0, cy0 = f0 * k, cx0 * k, cy0 * k
        cw = round(h0 * W / H); x0 = round(cx0 - cw / 2); x0 = min(max(x0, 0), w0 - cw)   # 比例不同（例：1200×1600）就以主點為中心裁成 9:16
        img = cv2.resize(img[:, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA)
        sc = W / cw; fx, cx, cy = f0 * sc, (cx0 - x0) * sc, cy0 * sc
        cv2.imwrite(str(out / 'img' / f'{n}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        d = depth(P, S, R, C, fx, cx, cy)
        m = pick.get('mask', {}).get(n)   # 前景有網格沒有的東西（例：大石頭），深度會變成後面的岩壁：那一塊留 0
        if m and 'below' in m: d[int(H * m['below']):] = 0
        cv2.imwrite(str(out / 'depth' / f'{n}.exr'), d)
        L = np.log(np.maximum(d, NEAR)); lo, hi = np.percentile(L[d > 0], (2, 98)) if (d > 0).any() else (0, 1)
        col = cv2.applyColorMap(np.clip((L - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO); col[d <= 0] = 0
        dl = cv2.dilate(L, np.ones((3, 3))) - cv2.erode(L, np.ones((3, 3)))   # 深度跳變（相鄰差超過 25%）＝輪廓，畫在照片上
        edge = cv2.dilate(((dl > 0.22) | ((d > 0) != (cv2.erode((d > 0).astype(np.uint8), np.ones((3, 3))) > 0))).astype(np.uint8), np.ones((2, 2))) > 0
        mix = img.copy(); mix[edge] = (0, 255, 255)
        cv2.imwrite(str(out / 'check' / f'{n}.jpg'), np.hstack([mix, cv2.addWeighted(img, 0.45, col, 0.55, 0)])[::2, ::2], [cv2.IMWRITE_JPEG_QUALITY, 85])
        cov = float((d > 0).mean())
        frames.append({'name': n, 'cover': round(cov, 3), 'position': np.round(C, 4).tolist(),
                       'quaternion': quat(np.column_stack([R[0], -R[1], -R[2]])),
                       'intrinsics': {'width': W, 'height': H, 'fx': round(fx, 4), 'fy': round(fx, 4), 'cx': round(cx, 2), 'cy': round(cy, 2)}})
        print(n, f'深度覆蓋 {cov:.0%}', flush=True)
        (out / 'frames.json').write_text(json.dumps(frames, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
