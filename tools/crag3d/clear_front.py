"""戶外 3D Topo：裁掉各區鏡頭和岩壁之間擋住畫面的樹叢、草（展示用的潑濺模型）。

各區鏡頭（snap.html 的 snaps.json：位置、看的中心）往岩壁看，鏡頭到看的中心距離 D；畫面範圍內（視角 FOV 外加 PAD 度）、
離鏡頭比 D − GAP 近、高出那裡的地面 ABOVE 公尺以上、而且是植物顏色（綠減藍超過 GREEN）的點丟掉（岩柱、石頭留著，例：新手牆正前方的岩柱）；
離岩壁 FLOAT_GAP 以外、高出地面、半透明（不透明度 < TRANS）的雜點不分顏色、路線附近也丟（地面本身留著，畫面下方才不會空白）。
地面高度：每 CELL 公尺一格，那一格不透明點的第 LOW 百分位。路線 KEEP 公尺內的點不動。
鏡頭沿岩壁滑動換區時經過的位置也算（相鄰兩區之間每 STEP 內插一個鏡頭）。

環境變數 STRONG=區名,…：那幾區（只在那一區自己的鏡頭、那一區路線左右範圍內）岩壁前方 STRONG_GAP 以外、高出地面 ABOVE 的點全裁（不分顏色；前面沒有要留的大石頭的區才用，例：下降面）。

用法：clear_front.py <輸入.spz> <snaps.json> <crag.json> <輸出.spz>
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read, write

GAP, ABOVE, CELL, LOW, KEEP, PAD, STEP, SOLID, GREEN, DARK = 1.2, 0.3, 0.5, 10, 1.0, 8, 0.25, 0.3, 0.04, 0.28
TRANS, FLOAT_GAP, STRONG_GAP = 0.6, 0.6, 1.5


def main():
    src, snaps, crag, out = sys.argv[1:]
    s = read(src); X = s['xyz']; n = len(X)
    snaps = json.loads(Path(snaps).read_text())
    cams = []
    for a, b in zip(snaps, snaps[1:] + [None]):
        ca = (np.array(a['camera']['position']), np.array(a['camera']['target']), a['camera']['fov'], a['wall'])
        cams.append(ca + (True,))
        if b:
            cb = (np.array(b['camera']['position']), np.array(b['camera']['target']), b['camera']['fov'], b['wall'])
            for t in np.arange(STEP, 1, STEP): cams.append((ca[0] + t * (cb[0] - ca[0]), ca[1] + t * (cb[1] - ca[1]), ca[2], ca[3] if t < 0.5 else cb[3], False))
    solid = s['alpha'] >= SOLID * 255
    key = np.floor(X[:, [0, 2]] / CELL).astype(np.int64); kk = key[:, 0] * 100003 + key[:, 1]
    o = np.argsort(kk[solid], kind='stable'); ks, ys = kk[solid][o], X[solid, 1][o]
    uk, start = np.unique(ks, return_index=True)
    gv = np.array([np.percentile(g, LOW) for g in np.split(ys, start[1:])])
    i = np.searchsorted(uk, kk); i = np.clip(i, 0, len(uk) - 1)
    gy = np.where(uk[i] == kk, gv[i], -np.inf)
    pts = np.array([p for r in json.loads(Path(crag).read_text())['routes'] for p in r.get('points') or []])
    near = cKDTree(pts).query(X, distance_upper_bound=KEEP)[0] < KEEP
    rgb = 0.5 + 0.28209479 * ((s['color'] / 255 - 0.5) / 0.15)   # SH0 → 顏色
    plant = (rgb[:, 1] - rgb[:, 2] > GREEN) | (rgb.mean(1) < DARK)   # 植物（綠、枯黃：綠減藍）和樹叢裡的陰影（很暗）；亮灰色的岩石、岩柱不動
    ghost = s['alpha'] < TRANS * 255   # 半透明（實心的岩石、石頭不會）
    drop = np.zeros(n, bool)
    strong = set(filter(None, os.environ.get('STRONG', '').split(',')))
    crag_j = json.loads(Path(crag).read_text()); span = {}
    for sn in snaps:
        if sn['wall'] not in strong: continue
        wl = next(x for x in crag_j['walls'] if x['name'] == sn['wall'])
        P = np.array([p for r in crag_j['routes'] if wl['routes'][0] <= int(r['number']) <= wl['routes'][1] for p in r.get('points') or []])
        C0, T0 = np.array(sn['camera']['position']), np.array(sn['camera']['target']); f0 = (T0 - C0) / np.linalg.norm(T0 - C0)
        sd = np.cross(f0, [0, 1, 0]); sd /= np.linalg.norm(sd); l = (P - T0) @ sd
        span[sn['wall']] = (l.min() - 0.5, l.max() + 0.5)
    for C, T, fov, w, own in cams:
        f = T - C; D = np.linalg.norm(f); f /= D
        v = X - C; z = v @ f
        cos = z / np.maximum(np.linalg.norm(v, axis=1), 1e-6)
        inview = (z > 0) & (cos > np.cos(np.radians(fov * 0.75 + PAD)))   # 直式畫面：對角大約是垂直視角的 0.75 倍
        drop |= inview & (z < D - GAP) & (X[:, 1] > gy + ABOVE) & plant & ~near
        drop |= inview & (z < D - FLOAT_GAP) & (X[:, 1] > gy + ABOVE) & ghost   # 飄在岩壁前面的半透明雜點：路線附近也裁
        if w in strong and w in span and own:   # 只在那一區自己的鏡頭、只裁那一區路線左右範圍內（換區途中的鏡頭也套用的話，會裁掉隔壁區真正的岩石，露出天空）
            side = np.cross(f, [0, 1, 0]); side /= np.linalg.norm(side); lat = (X - T) @ side; lo, hi = span[w]
            drop |= inview & (z < D - STRONG_GAP) & (X[:, 1] > gy + ABOVE) & (lat > lo) & (lat < hi)   # 指定的區：岩壁前方高出地面的全裁、不分顏色（例：下降面靠近鏡頭的乾草，畫面下半一大片半透明斜的）
    for k in ('xyz', 'alpha', 'color', 'logs', 'quat'): s[k] = s[k][~drop]
    write(out, s)
    print(f'{len(cams)} 個鏡頭位置；鏡頭前擋住的 {drop.sum()} 點丟掉，留下 {(~drop).sum()} → {out}（{Path(out).stat().st_size / 1e6:.1f} MB）')


if __name__ == '__main__':
    main()
