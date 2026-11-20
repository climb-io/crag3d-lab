"""戶外 3D Topo：預先算好每條路線從各個方向看會不會被岩壁擋住，手機只查表（不在手機上算深度）。

- 路線：化簡過的折線（跟 redpoint_pack.py 一樣，誤差 4 cm）每 STEP 公尺補一個點，再貼到看得到的岩面上；這些點就是輸出的折線，查表的順序跟著它。
- 貼到岩面：路線是畫在別的模型上的，岩縫、破洞的地方會埋在這個模型的岩面裡面。從那一區的鏡頭（snap.html 的 snaps.json）往每個點看，
  點前方 DRAPE 公尺內第一個碰到的岩面（DENSE 個點以上）就是看到的地方，點移到它前面一點；沒碰到就不動。
  點本身 ON_SURF 公尺內就有岩面（已經貼在岩面上）只在 ON_SURF 內微調：被前面的岩柱、岩塊擋住的那一段不能貼到擋住它的東西上
  （例：砲管 34 號上段被左邊的岩柱擋住，貼過去變 S 形）。路線在那一區鏡頭看不到的側面時（edit.drape_front，例：52 號在鳥頭岩塊朝右的側面）照舊貼到看得到的岩面。
  路線本來就是照那一區的正面畫的，鏡頭看過去第一個碰到的岩面就是畫路線的地方（大砲岩飛鳥 53 號埋在岩面後 4 m 多：舊的路線投影偏了）。
- 延伸到岩頂：頂繩路線畫到岩壁最上面（舊的路線照片常常只畫到一半）。從最後一點每 UP 公尺往上，沿那一區的朝向（crag.json 的 walls[].facing）
  在岩面前後 REACH 公尺內找岩面貼上去；找不到（過了岩頂）、岩面前後一步跳超過 JUMP 公尺（岩頂的草叢往外突出、岩頂往後退）、
  附近的點偏綠（植物：綠減藍超過 GREEN，SH0 換成的顏色）、或高過那一區的岩頂（walls[].top）TOP_PAD 公尺就停。不然線會一路畫進岩頂的草叢。
- 整理：放大看才看得出的怪線在這裡去掉（翻過岩頂往後走、深度突然跳開的點、往回折的點）。上面三成裡一步往水平方向走得比往上多 OVER 倍
  是翻過岩頂（例：下降面 19 號延伸時沿岩頂往後跑 2.5 m 又折回來；砲管 40 號沿岩頂往旁邊的岩塊走），從那裡截斷；沿朝向的深度跟前後 7 點的中位數差超過 SPIKE 公尺的點拉回中位數
  （描線時深度取錯的點，例：41、53 號）；比已經到過的高度低 BACK 公尺以上的點去掉。
- 分叉、合併：共用的一段在每條路線裡座標要完全相同（公開格式的規則），各條分開處理會差一點；最後照路線順序，
  每個點離前面路線的點 SHARE 公尺內就換成那個點。
- 方向：從路線點往相機看的方向，方位角每 360/AZ 度一格（atan2(x, z)，跟紅點貓 3D 的相機一樣），仰角從 EL0 每 EL_STEP 度一格共 EL_N 格。
- 擋住：高斯潑濺模型（spz_crop.py 的輸出）裡不透明度 ≥ SOLID 的點當岩壁（每 VOXEL 公尺留一個），
  沿著那個方向、離視線 RADIUS 公尺內、在路線點前方 EPS 公尺以外最近的一個點的距離。EPS 內的不算：路線貼著凹凸的岩面，斜著看會擋到旁邊的岩面，
  畫出來還是看得到；真正擋住路線的岩柱、轉角都在更遠的地方（用各區的鏡頭核對：EPS 0.4 m 時路線平均只看得到 47%，1.0 m 86%）。
  岩壁要有厚度：視線上 THICK 公尺內至少 DENSE 個點才算擋住，岩縫裡的草、飄著的葉子只有零星幾個點，不算。
  每個點每個方向一個 4 位元的值：0 是 DIST[-1] 公尺內都沒有擋住，k 是最近的擋住在 DIST[k-1] 公尺以內。
  看的人離路線點比那個距離遠就是被擋住。
- 輸出（JSON，路線編號 → { points, sight }）：sight 是 deflate（raw）壓過再 base64 的值，每個點 AZ×EL_N 個 4 位元，
  一個位元組放兩個（低 4 位元在前），點接著點；規格（SPEC）一起寫進去，紅點貓照它查。

一次只算一個方向（cKDTree 只建在畫面的二維座標上），記憶體只要幾百 MB。

用法：sight.py <模型.spz> <crag.json> <snaps.json> <輸出.json> [輸出 crag.json（路線換成延伸、貼面後的點，給快照重拍 2D 線）]
"""
import base64, json, sys, zlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read

STEP, SOLID, VOXEL, RADIUS, EPS, THICK, DENSE, DRAPE, ON_SURF = 0.4, 0.5, 0.05, 0.1, 1.2, 0.3, 4, 6.0, 0.3
UP, REACH, TOP_PAD, JUMP, GREEN, GREEN_RUN = 0.2, 1.5, 1.0, 0.3, 0.06, 3
GROUND_R, GROUND_PAD, BACK_STEP, BOULDER = 1.0, 0.3, 0.8, 0.6
OVER, SPIKE, BACK, SHARE = 1.2, 0.25, 0.05, 0.1
HOOK, LATERAL, BASE, STEP_D = 75, 1.0, 2.0, 1.0
ALIGN_DY, ALIGN_GAP = 0.6, 2.5
SLAB, CORNER_W, BIN, CORNER_D, GAP = 0.1, 2.0, 0.05, 0.15, 0.3
AZ, EL0, EL_STEP, EL_N = 36, -80, 10, 15
DIST = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
SPEC = {'step': STEP, 'az': AZ, 'el': [EL0, EL_STEP, EL_N], 'dist': DIST}


def simplify(a, tol):   # 3D 折線化簡（Douglas–Peucker），跟 redpoint_pack.py 同一個
    a = np.asarray(a)
    if len(a) < 3: return a.tolist()
    d = np.linalg.norm(np.cross(a[1:-1] - a[0], a[-1] - a[0]), axis=1) / max(np.linalg.norm(a[-1] - a[0]), 1e-9)
    k = int(d.argmax()) + 1
    return [a[0].tolist(), a[-1].tolist()] if d[k - 1] <= tol else simplify(a[:k + 1], tol)[:-1] + simplify(a[k:], tol)


def densify(pts):
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / STEP)))
        out += [a + (b - a) * (k + 1) / n for k in range(n)]
    return np.round(np.array(out), 2)


def main():
    spz, crag_file, snaps_file, out, *crag_out = sys.argv[1:]
    s = read(spz)
    solid = s['alpha'] >= SOLID * 255
    q, rgb = s['xyz'][solid], 0.5 + 0.28209479 * ((s['color'][solid] / 255 - 0.5) / 0.15)   # SH0 → 顏色
    _, first = np.unique(np.floor(q / VOXEL).astype(np.int64), axis=0, return_index=True)
    q, rgb = q[first], rgb[first]
    crag = json.loads(Path(crag_file).read_text())
    cams = {s['wall']: np.array(s['camera']['position']) for s in json.loads(Path(snaps_file).read_text())}
    wall_of = lambda num: next((w['name'] for w in crag['walls'] if w['routes'][0] <= int(num) <= w['routes'][1]), None)
    tree3 = cKDTree(q); tree2 = cKDTree(q[:, [0, 2]])
    def drape(num, pts):
        cam = cams.get(wall_of(num))
        if cam is None: return pts
        out = []
        for p in pts:
            d = cam - p; d /= np.linalg.norm(d)
            near = not edits.get(num, {}).get('drape_front') and tree3.query_ball_point(p, ON_SURF, return_length=True) >= DENSE   # 點已經在岩面上：只在附近微調，不往前貼到擋在前面的岩柱、岩塊上
            ts = np.arange(ON_SURF if near else DRAPE, -0.3, -0.05)
            hit = [t for t, n in zip(ts, tree3.query_ball_point(p + ts[:, None] * d, RADIUS, return_length=True)) if n >= DENSE]
            out.append(p + d * (hit[0] + 0.05) if hit else p)
        return np.round(np.array(out), 2)
    wall = lambda num: next(w for w in crag['walls'] if w['routes'][0] <= int(num) <= w['routes'][1])
    def trim_start(p, f):
        t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        while len(p) > 3 and abs(p[0] @ f - np.median(p[(t > t[0]) & (t <= t[0] + BASE)] @ f)) > JUMP:   # 起攀落在岩壁前的地面、大石頭
            p, t = p[1:], t[1:]
        low = np.nonzero(np.abs(np.diff(p @ f))[:max(1, int(0.3 * len(p)))] > STEP_D)[0]   # 好幾個點一起落在前面：下面三成裡一步前後跳超過 STEP_D，跳之前的都不要
        return p[low[-1] + 1:] if len(low) and len(p) - low[-1] - 1 > 3 else p
    def extend(num, pts):
        w = wall(num); f = np.array(w['facing']) if w.get('facing') else None
        ledge = edits.get(num, {}).get('ledge')
        if f is None or '橫渡' in names.get(num, ''): return pts   # 橫渡：橫著走，不往上下延伸
        ts = np.arange(REACH, -REACH, -0.05)
        def walk(p, step, stop):   # 沿岩面每 step 公尺往上（下）找岩面貼上去；連續 GREEN_RUN 步都是草才停（零星的苔不算），stop(p) 到了就停
            out, green = [], 0
            while not stop(p):
                base = p + [0, step, 0]
                hit = [t for t, n in zip(ts, tree3.query_ball_point(base + ts[:, None] * f, RADIUS, return_length=True)) if n >= DENSE]
                if hit and step < 0 and JUMP < hit[0] + 0.05 <= BOULDER:   # 往下走碰到前面凸出來的大石頭（BOULDER 公尺內）：從石頭上開始
                    out.append(base + f * (hit[0] + 0.05)); break
                if not hit or not -(JUMP if ledge else BACK_STEP) <= hit[0] + 0.05 <= JUMP: break   # 往前凸最多 JUMP（再多多半是前面的樹叢）；往後退（台階、岩頂往後收）最多 BACK_STEP，只到平台的路線（edit.ledge）碰到台階就停
                p = base + f * (hit[0] + 0.05)
                near = tree3.query_ball_point(p, 0.2)
                green = green + 1 if near and (rgb[near, 1] - rgb[near, 2]).mean() > GREEN else 0
                if green >= GREEN_RUN: out = out[:len(out) - GREEN_RUN + 1]; break
                out.append(p)
            return out
        pts = trim_start(pts, f * [1, 0, 1] / np.linalg.norm(f * [1, 0, 1]))   # 照片投下來的起攀先整理，再往下接（不然往下接的點會被當成落在前面刪掉）
        top = w['top'][1] + TOP_PAD
        up = walk(pts[-1], UP, lambda p: p[1] >= top)
        # 起攀往下接到地面：附近 GROUND_R 公尺內模型最低的點以上 GROUND_PAD 公尺（舊照片的紅線常從半高的台階開始畫）
        near = tree2.query_ball_point(pts[0][[0, 2]], GROUND_R)
        ground = q[near, 1].min() + GROUND_PAD if near else pts[0][1]
        down = walk(pts[0], -UP, lambda p: p[1] <= ground)
        pts = np.concatenate([np.round(down[::-1][::2], 2), pts]) if len(down) else pts
        return np.concatenate([pts, np.round(up[1::2], 2)]) if len(up) > 1 else pts
    # 路線名稱是明確的定位點：名稱有「內角」「煙囪」的路線在凹進去的角裡，有「岩角」的在凸出的稜上。
    # 每個點在自己的高度切一片（±SLAB 公尺、左右 ±CORNER_W 公尺），沿岩壁左右每 BIN 公尺取最前面的岩面當剖面，
    # 剖面上兩邊都比它前面（凹角）或後面（岩角）CORNER_D 公尺以上的那一格就是角；找得到才移過去，移的量沿線取中位數，不會一點一點跳
    names = {r['number']: r.get('name') or '' for r in crag['routes']}
    edits = {r['number']: r.get('edit') or {} for r in crag['routes']}   # 使用者的指正（route_edits.py）：ledge、trim_top、no_anchor
    def anchor(num, pts):
        name = names.get(num, '')
        sign = 0 if edits.get(num, {}).get('no_anchor') else -1 if ('內角' in name or '煙囪' in name) else 1 if '岩角' in name else 0
        if not sign: return pts
        f = np.array(wall(num).get('facing') or [0, 0, 1], float); f[1] = 0; f /= np.linalg.norm(f)
        side = np.cross([0, 1, 0], f); side /= np.linalg.norm(side)
        shift = np.full(len(pts), np.nan)
        for i, p in enumerate(pts):
            near = np.array(tree3.query_ball_point(p, CORNER_W + 0.2))
            if not len(near): continue
            v = q[near] - p
            v = v[np.abs(v[:, 1]) <= SLAB]
            a, d = v @ side, v @ f
            k = np.floor((a + CORNER_W) / BIN).astype(int); ok = (k >= 0) & (k < int(2 * CORNER_W / BIN))
            prof = np.full(int(2 * CORNER_W / BIN), np.nan)
            for kk, dd in zip(k[ok], d[ok]): prof[kk] = dd if np.isnan(prof[kk]) else max(prof[kk], dd)
            best = None
            for j in range(len(prof)):
                if np.isnan(prof[j]): continue
                left, right = prof[:j], prof[j + 1:]
                if not (np.any(~np.isnan(left)) and np.any(~np.isnan(right))): continue
                lo, ro = (np.nanmax(left), np.nanmax(right)) if sign < 0 else (np.nanmin(left), np.nanmin(right))
                if sign < 0 and lo - prof[j] >= CORNER_D and ro - prof[j] >= CORNER_D or sign > 0 and prof[j] - lo >= CORNER_D and prof[j] - ro >= CORNER_D:
                    off = (j + 0.5) * BIN - CORNER_W
                    if best is None or abs(off) < abs(best[0]): best = (off, prof[j])
            if best: shift[i] = best[0]
        if np.all(np.isnan(shift)): return pts
        shift = np.where(np.isnan(shift), np.nanmedian(shift), shift)
        shift = np.array([np.median(shift[max(0, i - 3):i + 4]) for i in range(len(shift))])
        return drape(num, pts + shift[:, None] * side)
    def tidy(num, pts):
        f = np.array(wall(num).get('facing') or [0, 0, 1], float); f[1] = 0; f /= np.linalg.norm(f)
        if '橫渡' in names.get(num, ''): return np.round(pts, 2)   # 橫渡：橫著走，往回折、翻過岩頂、示意圖的規則都是給往上爬的路線
        d, y = pts @ f, pts[:, 1]
        lo, hi = y.min(), y.max()
        top_y = wall(num)['top'][1]
        keep = pts[:, 1] <= top_y + TOP_PAD   # 超過岩頂的（照片紅線的頂端碰到樹枝、草叢）不要
        pts = pts[keep] if keep.sum() >= 2 else pts
        d, y = pts @ f, pts[:, 1]
        for i in range(1, len(pts)):   # 翻過岩頂：岩頂 0.5 m 內，一步往水平方向（前後或左右）走得比往上多 OVER 倍
            step = pts[i] - pts[i - 1]; back = min(0, step @ f)   # 往岩壁裡退（台階、岩頂往後收）的那一段不算翻過岩頂
            dd, dy = np.hypot(*(step - back * f)[[0, 2]]), y[i] - y[i - 1]
            if y[i] > top_y - 0.5 and dd > 0.15 and dd > OVER * max(dy, 0): pts, d, y = pts[:i], d[:i], y[:i]; break
        pts = pts.copy()
        for i in range(len(pts)):   # 深度（沿岩壁朝向）突然跳開的點：拉回前後 7 點的中位數
            m = np.median(d[max(0, i - 3):i + 4])
            if abs(d[i] - m) > SPIKE: pts[i] += f * (m - d[i])
        top, keep = -np.inf, []
        for i, p in enumerate(pts):   # 往回折的點（比已經到的高度低）不要
            if p[1] >= top - BACK: keep.append(i); top = max(top, p[1])
        p = smooth(pts[keep], f); e = edits.get(num, {})
        if e.get('ledge'):   # 只到上方的平台：上半段第一個往岩壁裡退超過 JUMP 的台階就截掉（照片紅線常畫過平台）
            back = np.nonzero((np.diff(p @ f) < -JUMP) & (p[1:, 1] > p[:, 1].min() + 0.5 * np.ptp(p[:, 1])))[0]
            if len(back): p = p[:back[0] + 1]
        if e.get('trim_top'): p = p[p[:, 1] <= p[:, 1].max() - e['trim_top']] if (p[:, 1] <= p[:, 1].max() - e['trim_top']).sum() >= 2 else p
        return np.round(p, 2)

    # 路線是示意圖（看得出往哪爬就好），不能彎來彎去：起攀前段離後面 BASE 公尺內的岩面超過 JUMP 公尺的點（地面、大石頭）去掉、結尾的小勾去掉（起攀在延伸前整理過）、沿岩壁的左右位置平滑（前後、高度不動，還是貼著岩壁）、
    # 剩下轉角超過 HOOK 度的點去掉。舊照片上本來就有的大轉彎（例：沿岩角繞）平滑後還在，只是變圓順
    def turn(p, i):
        u, v = p[i] - p[i - 1], p[i + 1] - p[i]
        if np.linalg.norm(u) < 0.03 or np.linalg.norm(v) < 0.03: return 0
        return np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    def smooth(p, f):
        while len(p) > 3 and turn(p, len(p) - 2) > HOOK: p = p[:-1]
        side = np.cross([0, 1, 0], f); side /= np.linalg.norm(side)
        t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        w = np.exp(-0.5 * ((t[:, None] - t[None]) / LATERAL) ** 2)
        a = p @ side; p = p + ((w @ a) / w.sum(1) - a)[:, None] * side
        i = 1
        while i < len(p) - 1 and len(p) > 3:
            if turn(p, i) > HOOK: p = np.delete(p, i, 0); i = max(1, i - 1)
            else: i += 1
        return p
    routes = [(r['number'], tidy(r['number'], anchor(r['number'], extend(r['number'], drape(r['number'], densify(np.array(simplify(r['points'], 0.04)))))))) for r in crag['routes'] if r.get('points')]
    # 路線不越過內角：同一區裡名稱有「內角」「煙囪」的路線定位好之後，其他路線在每個高度跟它比左右，
    # 大部分在哪一側就留在哪一側，跑到另一側的點拉回這一側、離內角 GAP 公尺，再貼面（例：無名面右線往上爬滑到下降面）
    corner = {num for num, _ in routes if '內角' in names.get(num, '') or '煙囪' in names.get(num, '')}
    for k, (num, p) in enumerate(routes):
        if num in corner: continue
        f = np.array(wall(num).get('facing') or [0, 0, 1], float); f[1] = 0; f /= np.linalg.norm(f)
        side = np.cross([0, 1, 0], f); side /= np.linalg.norm(side)
        for c in corner:
            if wall_of(c) != wall_of(num): continue
            cp = dict(routes)[c]; o = np.argsort(cp[:, 1])
            ac = np.interp(p[:, 1], cp[o, 1], cp[o] @ side, left=np.nan, right=np.nan)
            rel = p @ side - ac; ok = ~np.isnan(rel)
            if not ok.any(): continue
            sgn = np.sign(np.median(rel[ok])); bad = ok & (rel * sgn < GAP)
            if bad.any() and abs(np.median(rel[ok])) < 3 * CORNER_W:
                p = p.copy(); p[bad] += ((ac[bad] + sgn * GAP) - p[bad] @ side)[:, None] * side
                p = tidy(num, drape(num, p))
        routes[k] = (num, p)
    # 起點、終點的高度照指定（edit.span = [起點高度, 終點高度]，例：換模型展示時沿用上一版確認過的）：多的截掉，不夠的每 STEP 往下（上）沿那一區的朝向找岩面補上
    def snapf(base, f):
        ts = np.arange(REACH, -REACH, -0.05)
        hit = [t for t, n in zip(ts, tree3.query_ball_point(base + ts[:, None] * f, RADIUS, return_length=True)) if n >= DENSE and abs(t) <= JUMP]
        return base + f * (min(hit, key=abs) + 0.05) if hit else base   # 只貼最近的岩面（前後 JUMP 內），不跳到前面的大石頭上
    for k, (num, p) in enumerate(routes):
        sp = edits.get(num, {}).get('span')
        if not sp: continue
        f = np.array(wall(num).get('facing') or [0, 0, 1], float); f[1] = 0; f /= np.linalg.norm(f)
        y0, y1 = sp; m = (p[:, 1] >= y0 - 0.05) & (p[:, 1] <= y1 + 0.05)
        if m.sum() >= 2: p = p[m]
        while p[0, 1] > y0 + 0.1: p = np.vstack([snapf(p[0] + [0, -min(STEP, p[0, 1] - y0), 0], f), p])
        while p[-1, 1] < y1 - 0.1: p = np.vstack([p, snapf(p[-1] + [0, min(STEP, y1 - p[-1, 1]), 0], f)])
        routes[k] = (num, np.round(p, 2))
    # 相鄰路線的起點對齊：同一區左右相鄰（沿岩壁 ALIGN_GAP 公尺內）、起點高度差 ALIGN_DY 以內的一群，起點都截到那一群最高的起點
    # （只截短、不往下補進地面），號碼牌排成一條水平線比較整齊（使用者 2026-11-17）
    got = dict(routes)
    for w in crag['walls']:
        nums = [n for n, _ in routes if w['routes'][0] <= int(n) <= w['routes'][1] and '橫渡' not in names.get(n, '')]
        if len(nums) < 2: continue
        f = np.array(w.get('facing') or [0, 0, 1], float); f[1] = 0; f /= np.linalg.norm(f); side = np.cross([0, 1, 0], f)
        nums.sort(key=lambda n: got[n][0] @ side)
        groups, g = [], [nums[0]]
        for a, b in zip(nums, nums[1:]):
            if abs(got[a][0][1] - got[b][0][1]) <= ALIGN_DY and abs((got[a][0] - got[b][0]) @ side) <= ALIGN_GAP: g.append(b)
            else: groups.append(g); g = [b]
        groups.append(g)
        for g in groups:
            if len(g) < 2: continue
            top = max(got[n][0][1] for n in g)
            for n in g:
                p = got[n]
                if p[0][1] >= top - 0.02: continue
                k = int(np.argmax(p[:, 1] >= top))
                if k == 0 or p[k, 1] < top: continue
                a, b = p[k - 1], p[k]; t = (top - a[1]) / max(1e-6, b[1] - a[1])
                got[n] = np.round(np.vstack([a + t * (b - a), p[k:]]), 2)
    routes = [(n, got[n]) for n, _ in routes]
    # 分叉、合併（公開格式的規則）：共用的一段在每條路線裡座標要完全相同。各條分開貼面、整理會差一點，
    # 所以照路線順序，每個點離前面路線的點 SHARE 公尺內就換成那個點（紅點貓靠座標相同認共用段，只畫一次）
    seen = np.empty((0, 3))
    for k, (num, p) in enumerate(routes):
        if len(seen):
            d, j = cKDTree(seen).query(p)
            p = np.where((d <= SHARE)[:, None], seen[j], p)
            routes[k] = (num, p)
        seen = np.concatenate([seen, p])
    if crag_out:
        got = dict(routes)
        for r in crag['routes']:
            if r['number'] in got: r['points'] = got[r['number']].tolist()
        Path(crag_out[0]).write_text(json.dumps(crag, ensure_ascii=False))
    P = np.concatenate([p for _, p in routes])
    print(f'岩壁 {len(q)} 點，路線 {len(routes)} 條 {len(P)} 點')
    edges = np.array(DIST)
    val = np.zeros((len(P), EL_N, AZ), np.uint8)
    for e in range(EL_N):
        el = np.radians(EL0 + e * EL_STEP)
        for a in range(AZ):
            az = np.radians(a * 360 / AZ)
            d = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
            u = np.cross(d, [0, 1, 0] if abs(d[1]) < 0.9 else [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(d, u)
            B = np.stack([u, v], 1)
            tree = cKDTree(q @ B)
            dq, dp = q @ d, P @ d
            for i, near in enumerate(tree.query_ball_point(P @ B, RADIUS)):
                if not near: continue
                rel = np.sort(dq[near] - dp[i])
                rel = rel[rel > EPS]
                solid = np.nonzero(rel[DENSE - 1:] - rel[:len(rel) - DENSE + 1] < THICK)[0] if len(rel) >= DENSE else []
                if len(solid): val[i, e, a] = min(np.searchsorted(edges, rel[solid[0]]) + 1, len(DIST) + 1) % (len(DIST) + 1)
        print(f'仰角 {EL0 + e * EL_STEP}° 完成', flush=True)
    res, i = {}, 0
    for num, p in routes:
        v = val[i:i + len(p)].reshape(-1); i += len(p)
        if len(v) % 2: v = np.append(v, 0)
        packed = (v[0::2] | (v[1::2] << 4)).astype(np.uint8).tobytes()
        c = zlib.compressobj(9, zlib.DEFLATED, -15)
        res[num] = {'points': p.tolist(), 'sight': base64.b64encode(c.compress(packed) + c.flush()).decode()}
    Path(out).write_text(json.dumps({'spec': SPEC, 'routes': res}, separators=(',', ':')))
    blocked = (val > 0).mean()
    print(f'有擋住的方向 {blocked:.0%}，{Path(out).stat().st_size / 1e3:.0f} kB → {out}')


if __name__ == '__main__':
    main()
