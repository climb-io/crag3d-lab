"""戶外 3D Topo：在某一區的正面快照上描的路線（像素折線）打到高斯潑濺模型上，換掉那幾條路線的 3D 座標。

舊路線照片跟潑濺模型的畫面自動配對不起來時用（差 12 年、光線不同，潑濺近看又糊，SIFT 只配到十幾個點）：
照舊照片上路線跟裂縫、岩角、岩塊邊緣的相對位置，在 snap.html 拍的正面快照上描出來（由下往上），
快照的鏡頭已知，深度用模型裡不透明的點畫 z-buffer，每個像素就是一個 3D 點。深度沿線取 SMOOTH 點的中位數：
經過草叢的地方深度忽前忽後，不平滑的話換個角度看是鋸齒。

描線檔（JSON）：{ "<區名>": { "<路線編號>": [[x, y], …] } }，座標是快照 wall-<序號>.jpg 的像素（snaps.json 的 w、h）。
用法：route_draw.py <模型.spz> <crag.json> <snaps 資料夾> <描線檔> <輸出 crag.json> [核對圖資料夾]
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.ndimage import grey_erosion
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read

SOLID, FILL, STEP_PX, SMOOTH, FAR_Z = 0.5, 5, 6, 9, 2.0

spz, crag_file, snaps_dir, drawn_file, out_file, *check = sys.argv[1:]
snaps_dir = Path(snaps_dir)
s = read(spz); Q = s['xyz'][s['alpha'] >= SOLID * 255]
crag = json.loads(Path(crag_file).read_text())
drawn = json.loads(Path(drawn_file).read_text())
snaps = json.loads((snaps_dir / 'snaps.json').read_text())


def look(cam):
    C, T = np.array(cam['position'], float), np.array(cam['target'], float)
    f = (T - C) / np.linalg.norm(T - C); r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    return np.stack([r, -u, f]), C


def depth_map(R, C, K, w, h):
    X = (Q - C) @ R.T
    X = X[X[:, 2] > 0.3]
    uv = X[:, :2] / X[:, 2:] @ K[:2, :2].T + K[:2, 2]
    i, j = np.round(uv[:, 1]).astype(int), np.round(uv[:, 0]).astype(int)
    ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    d = np.full(h * w, np.inf); np.minimum.at(d, i[ok] * w + j[ok], X[ok, 2])
    d = d.reshape(h, w)
    filled = grey_erosion(np.where(np.isinf(d), 1e9, d), size=(FILL, FILL))
    return np.where(np.isinf(d), np.where(filled < 1e9, filled, np.inf), np.minimum(d, filled + 0.15))


def dense(line):   # 像素折線每 STEP_PX 補一點
    out = [line[0]]
    for a, b in zip(line, line[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / STEP_PX)))
        out += [a + (b - a) * (k + 1) / n for k in range(n)]
    return np.array(out)


for wall, lines in drawn.items():
    i, sn = next((i, x) for i, x in enumerate(snaps) if x['wall'] == wall)
    w, h = sn['w'], sn['h']; fy = h / 2 / np.tan(np.radians(sn['camera']['fov']) / 2)
    K = np.array([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]]); R, C = look(sn['camera'])
    D = depth_map(R, C, K, w, h)
    img = cv2.imread(str(snaps_dir / sn['file'])) if check else None
    for num, line in lines.items():
        uv = dense(np.array(line, float))
        z = D[np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1), np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)]
        ok = np.isfinite(z)
        zs = z.copy(); h2 = SMOOTH // 2
        for k in np.nonzero(ok)[0]:
            win = z[max(0, k - h2):k + h2 + 1]; zs[k] = np.median(win[np.isfinite(win)])
        z = zs
        ok &= np.abs(z - np.median(z[ok])) < FAR_Z   # 描到岩壁邊緣外（後面遠處的岩石、地面）的點不要：深度跟整條的中位數差超過 FAR_Z 公尺
        X = C + ((np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(K).T) * z[:, None]) @ R
        route = next(r for r in crag['routes'] if r['number'] == num)
        old = np.array(route.get('points') or [])
        route['points'] = X[ok].round(3).tolist()
        print(wall, num, f'{ok.mean():.0%} 打到模型' + (f'，跟原本差 {np.linalg.norm(X[ok].mean(0) - old.mean(0)):.2f} m' if len(old) else '（新的路線）'))
        if img is not None:
            cv2.polylines(img, [uv.astype(np.int32)], False, (40, 40, 230), 7)
            cv2.putText(img, num, tuple(uv[0].astype(int) + [-10, 40]), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
    if img is not None:
        Path(check[0]).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(check[0]) / f'drawn-{i}.jpg'), cv2.resize(img, (w // 2, h // 2)))
Path(out_file).write_text(json.dumps(crag, ensure_ascii=False))
