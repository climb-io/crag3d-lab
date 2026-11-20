"""戶外 3D Topo：高斯潑濺模型（.spz v2、只有 SH0）轉到路線的座標，裁出看路線需要的那段岩壁。

- 座標：給了 align.json 就套它的相似變換（scale_atlas_to_holdlab、R_atlas_to_holdlab、t_atlas_to_holdlab），位置、旋轉、大小一起轉；- 表示已經在路線的座標。
- 範圍：照路線點定一個沿著岩壁的框。沿岩壁方向是全部路線點水平分布的主軸，
  只留第一條到最後一條路線往外 SIDE 公尺（--to 只到那個岩壁，之後的不要）；
  每 BIN 公尺一格，岩面的深度是附近路線點的中位數，留岩面前方 FRONT（起攀的地面）到後方 BEHIND；
  高度從附近最低的路線點往下 BELOW 到最高點往上 ABOVE。
  格子之間線性內插，框的邊緣跟著岩壁轉、不會一格一格。
- 邊緣不硬切：離框的邊 FADE 公尺內不透明度漸漸變 0（smoothstep），沿岩壁兩端、上下、前後都淡出，看起來像霧裡的岩壁、不像被刀切。
  路線 KEEP 公尺內的點不淡化、不裁掉（岩角、突出的岩塊會凸出框的前緣）。
- 雜點：幾乎透明（不透明度 < MIN_ALPHA）或特別大（最長軸 > MAX_SCALE 公尺）的點丟掉，多半是拍到的樹葉、天空。
- 大小（開放資料集每個檔案 ≤ 25 MB）：點照空間順序（每 CELL 公尺一格的 Morton 碼）排好，座標存到 1/2^FRAC 公尺，gzip 壓得比較小。

用法：spz_crop.py <輸入.spz> <align.json 或 -> <crag.json> <輸出.spz> [--to 岩壁名稱]
"""
import gzip, json, os, struct, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

SIDE, BIN, FRONT, BEHIND, BELOW, ABOVE = (float(os.environ.get(k, v)) for k, v in
                                          (('SIDE', 1.5), ('BIN', 1.0), ('FRONT', 2.5), ('BEHIND', 5.0), ('BELOW', 4.0), ('ABOVE', 5.0)))   # 環境變數可以放大（例：只看 3D 效果、畫面要滿版）
NEAR, FADE = 3.0, 2.5
MAX_SCALE = float(os.environ.get('MAX_SCALE', 1.5))
ROCK_NEAR = 4.0
MIN_ALPHA = float(os.environ.get('MIN_ALPHA', 0.16))   # 檔案太大（開放資料集每個檔案 ≤ 25 MB）時調高
FRAC, CELL, KEEP = 9, 0.02, 1.5


def read(path):
    d = gzip.open(path).read()
    magic, ver, n, sh, frac, flags, _ = struct.unpack('<4sIIBBBB', d[:16])
    if magic != b'NGSP' or ver != 2 or sh != 0: raise SystemExit(f'只讀 spz v2、SH0：{magic} v{ver} sh{sh}')
    o = 16
    def take(k):
        nonlocal o
        a = np.frombuffer(d, np.uint8, n * k, o).reshape(n, k); o += n * k
        return a
    pos, alpha, color, scale, rot = take(9), take(1)[:, 0], take(3), take(3), take(3)
    p = pos.reshape(n, 3, 3).astype(np.int32)
    v = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
    v = np.where(v & 0x800000, v - 0x1000000, v)
    return dict(n=n, frac=frac, flags=flags, xyz=v / (1 << frac), alpha=alpha, color=color,
                logs=scale / 16 - 10, quat=xyzw(rot))


def xyzw(rot):
    q = rot / 127.5 - 1
    return np.column_stack([q, np.sqrt(np.clip(1 - (q ** 2).sum(1), 0, 1))])


def write(path, s):
    n = len(s['xyz'])
    v = np.round(s['xyz'] * (1 << s['frac'])).astype(np.int64)
    if np.abs(v).max() >= 1 << 23: raise SystemExit('位置超出 24 位元的範圍')
    v &= 0xFFFFFF
    pos = np.stack([v & 255, (v >> 8) & 255, v >> 16], -1).astype(np.uint8).reshape(n, 9)
    q = s['quat'] * np.where(s['quat'][:, 3:] < 0, -1, 1)
    rot = np.clip(np.round((q[:, :3] + 1) * 127.5), 0, 255).astype(np.uint8)
    scale = np.clip(np.round((s['logs'] + 10) * 16), 0, 255).astype(np.uint8)
    body = b''.join(a.tobytes() for a in (pos, s['alpha'].astype(np.uint8), s['color'], scale, rot))
    Path(path).write_bytes(gzip.compress(struct.pack('<4sIIBBBB', b'NGSP', 2, n, 0, s['frac'], s['flags'], 0) + body, 9))


def qmul(a, b):
    ax, ay, az, aw = a.T; bx, by, bz, bw = b.T
    return np.column_stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def mat2quat(R):
    w = np.sqrt(max(0, 1 + np.trace(R))) / 2
    return np.array([(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w])


def transform(s, al):
    k, R, t = al['scale_atlas_to_holdlab'], np.array(al['R_atlas_to_holdlab']), np.array(al['t_atlas_to_holdlab'])
    if abs(np.linalg.det(R) - 1) > 1e-3: raise SystemExit('對齊的旋轉不是正規旋轉')
    s['xyz'] = k * s['xyz'] @ R.T + t
    s['quat'] = qmul(np.tile(mat2quat(R), (s['n'], 1)), s['quat'])
    s['logs'] = s['logs'] + np.log(k)


def main():
    args = sys.argv[1:]
    to = None
    if '--to' in args: i = args.index('--to'); to = args[i + 1]; del args[i:i + 2]
    src, align, crag_file, out = args
    s = read(src)
    if align != '-': transform(s, json.loads(Path(align).read_text()))
    crag = json.loads(Path(crag_file).read_text())
    walls = crag['walls']
    if to: walls = walls[:[w['name'] for w in walls].index(to) + 1]
    nums = {n for w in walls for n in range(w['routes'][0], w['routes'][1] + 1)}
    pts = np.array([p for r in crag['routes'] if int(r['number']) in nums for p in r['points']])
    stand = np.array([w['standAt'] for w in walls])
    c0 = pts[:, [0, 2]].mean(0)
    along = np.linalg.svd(pts[:, [0, 2]] - c0, full_matrices=False)[2][0]
    front = np.array([-along[1], along[0]])
    if ((stand[:, [0, 2]] - c0) @ front).mean() < 0: front = -front
    a_of = lambda xz: (xz - c0) @ along
    d_of = lambda xz: (xz - c0) @ front
    pa, pd = a_of(pts[:, [0, 2]]), d_of(pts[:, [0, 2]])
    lo, hi = pa.min() - SIDE, pa.max() + SIDE
    grid = np.arange(lo, hi + BIN, BIN)
    face, bottom, top = [], [], []
    for g in grid:
        m = np.abs(pa - g) <= max(BIN, NEAR)
        if not m.any(): m = np.abs(pa - g) <= np.sort(np.abs(pa - g))[min(20, len(pa) - 1)]
        face.append(np.median(pd[m])); bottom.append(pts[m, 1].min()); top.append(pts[m, 1].max())
    xa, xd = a_of(s['xyz'][:, [0, 2]]), d_of(s['xyz'][:, [0, 2]])
    f, b, tp = (np.interp(xa, grid, v) for v in (face, bottom, top))
    alpha = s['alpha'] / 255
    keep = ((xa >= lo) & (xa <= hi) & (xd <= f + FRONT) & (xd >= f - BEHIND)
            & (s['xyz'][:, 1] >= b - BELOW) & (s['xyz'][:, 1] <= tp + ABOVE)
            & (alpha >= MIN_ALPHA) & (np.exp(s['logs']).max(1) <= MAX_SCALE))
    y = s['xyz'][:, 1]
    inner = np.minimum.reduce([xa - lo, hi - xa, f + FRONT - xd, xd - (f - BEHIND), y - (b - BELOW), tp + ABOVE - y])
    u = np.clip(inner / FADE, 0, 1)
    # 路線 KEEP 公尺內的點不淡化：岩角、突出的岩塊會凸出框的前緣（例：梁大隆右岩角 11 號，原本 3.2 萬點變半透明、看起來白白的）
    near = cKDTree(pts).query(s['xyz'], distance_upper_bound=KEEP)[0] < KEEP
    u[near] = 1
    keep |= near & (alpha >= MIN_ALPHA)
    s['alpha'] = np.round(s['alpha'] * u * u * (3 - 2 * u))
    keep &= s['alpha'] >= MIN_ALPHA * 255
    cap = int(os.environ.get('MAX_POINTS', 0))   # 點數上限（檔案大小、手機效能）：離路線 KEEP 外、最不明顯的點先丟（不透明度 × 最大兩軸的面積；只看大小會丟掉地面、前景的細節，變斑駁）
    if cap and keep.sum() > cap:
        sc = np.sort(np.exp(s['logs']), 1); vol = s['alpha'] / 255 * sc[:, 1] * sc[:, 2]
        vol[cKDTree(pts).query(s['xyz'], distance_upper_bound=ROCK_NEAR)[0] < ROCK_NEAR] *= 100   # 路線 ROCK_NEAR 內的岩壁幾乎不丟（又小又半透明的點疊起來才是岩面，丟了會破洞）；先丟背景、樹林
        vol[near] = np.inf; vol[~keep] = -1
        drop = np.argsort(vol)[(~keep).sum():(~keep).sum() + keep.sum() - cap]; keep[drop] = False
    for key in ('xyz', 'alpha', 'color', 'logs', 'quat'): s[key] = s[key][keep]
    q = np.floor((s['xyz'] - s['xyz'].min(0)) / CELL).astype(np.int64); code = np.zeros(len(q), np.int64)
    for b in range(16):
        for a in range(3): code |= ((q[:, a] >> b) & 1) << (3 * b + a)
    order = np.argsort(code)
    for key in ('xyz', 'alpha', 'color', 'logs', 'quat'): s[key] = s[key][order]
    s['frac'] = FRAC
    write(out, s)
    print(f'{keep.sum()}／{len(keep)} 點，沿岩壁 {hi - lo:.1f} m，{Path(out).stat().st_size / 1e6:.1f} MB → {out}')


if __name__ == '__main__':
    main()
