"""戶外 3D Topo：舊路線照片投到模型上的路線（photo_pose.py batch 的輸出）換進 crag.json。

拍攝位置可信（PnP 內點 ≥ MIN_INLIERS、佔配對點 ≥ MIN_RATIO；例：梁大隆 6–9 的底圖內點只佔 18%，投出來的線飄到天上）、紅線打到模型的比例 ≥ MIN_SHARE 的路線才換；其他的留原本的點，列出來。
拍攝位置前方 NEAR 公尺內被模型擋住超過 MAX_BLOCK 的（相機跑進樹叢、岩塊裡，例：梁大隆 6–9）也不換。
點沿線每 STEP 公尺取一個（投線是紅線每 6 像素一點，遠近不一），再沿線做高斯平滑（SIGMA 公尺，頭尾固定）：
紅線描出來的位置、讀到的深度每一點都有一點抖動，不順會變成鋸齒。
平滑前先整理前後（從拍攝位置看過去的水平方向）：
- 起攀：舊照片的紅線下端常畫到岩壁前的地面、大石頭，前段離後面 BASE 公尺內的中位數超過 OFF 公尺的點去掉，從岩壁底開始。
- 中間：離前後 WIN 點滑動中位數超過 OFF 公尺的點，前後方向拉回中位數（打到岩壁前的草、凸起）。

用法：photo_routes.py <crag.json> <photo_pose 輸出資料夾> <輸出 crag.json> [模型.spz（檢查拍攝位置前方有沒有被擋）]
"""
import json, sys
from pathlib import Path
import numpy as np

MIN_INLIERS, MIN_RATIO, MIN_SHARE, STEP, SIGMA = 300, 0.18, 0.6, 0.25, 0.5
BASE, OFF, WIN = 2.0, 0.3, 7
NEAR, MAX_BLOCK = 1.5, 0.1


def resample(p):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0, d[-1], max(2, int(d[-1] / STEP) + 1))
    q = np.stack([np.interp(t, d, p[:, k]) for k in range(3)], 1)
    i = np.arange(len(q)); w = np.exp(-0.5 * ((i[:, None] - i[None]) * STEP / SIGMA) ** 2)
    out = w @ q / w.sum(1, keepdims=True); out[[0, -1]] = q[[0, -1]]
    return out


def tidy(p, cam):
    f = p.mean(0) - cam; f[1] = 0; f /= np.linalg.norm(f)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    while len(p) > 2 and abs(p[0] @ f - np.median(p[(d > d[0]) & (d <= d[0] + BASE)] @ f)) > OFF:
        p, d = p[1:], d[1:]
    fd = p @ f
    med = np.array([np.median(fd[max(0, i - WIN // 2):i + WIN // 2 + 1]) for i in range(len(fd))])
    off = np.abs(fd - med) > OFF
    return p + np.where(off, med - fd, 0)[:, None] * f


def blocked(Q, res):
    sys.path.insert(0, str(Path(__file__).parent))
    from photo_pose import depth_map, K_of
    D = depth_map(Q, np.array(res['R']), np.array(res['position']), K_of(res['fov'], 150, 200), 150, 200, 0.05, clamp=False)
    return float((D < NEAR).mean())


def main():
    crag_file, photo_dir, out, *spz = sys.argv[1:]
    c = json.loads(Path(crag_file).read_text())
    Q = None
    if spz:
        sys.path.insert(0, str(Path(__file__).parent))
        from spz_crop import read
        s = read(spz[0]); Q = s['xyz'][s['alpha'] >= 128]
    used, kept = [], []
    for r in c['routes']:
        f = Path(photo_dir) / f"{r['number']}.json"
        res = json.loads(f.read_text()) if f.exists() else {}
        if res.get('points') and res.get('inliers', 0) >= MIN_INLIERS and res['inliers'] >= MIN_RATIO * res.get('matches', 1e9) and (res.get('share') or 0) >= MIN_SHARE and (Q is None or 'R' not in res or blocked(Q, res) <= MAX_BLOCK):
            r['points'] = resample(tidy(resample(np.array(res['points'])), np.array(res['position']))).round(3).tolist(); used.append(r['number'])
        else: kept.append(r['number'])
    Path(out).write_text(json.dumps(c, ensure_ascii=False))
    print(f'照片投線 {len(used)} 條；留原本的 {len(kept)} 條：{" ".join(kept)}')


if __name__ == '__main__':
    main()
