"""戶外 3D Topo：挑好的 Atlas 照片（atlas64.json）哪裡缺照片、從有位置的照片裡挑哪幾張補（實驗）。

模型的點（每 STEP 個取一個）對每台相機做 z-buffer（1/DS 解析度，不做光線測試），沒被擋、斜角小於 70°（cos > MIN_COS）、
距離 MAX_D 公尺內的點算「好角度拍到」。每個點目標 NEED 張；路線 NEAR 公尺內的點權重 ROUTE_W 倍。

挑法（換照片，總數不變）：每輪加一張「補上最多缺口」的候選照片，再拿掉一張「拿掉後損失最少」的照片；
加的比拿掉的多 MIN_GAIN 倍才換。不加：禁用清單、使用者移除過的、跟已選照片太像的（位置 SAME_D 公尺內且方向 SAME_DEG 度內）。
不拿掉：必選（must）、使用者加的、使用者要加多的分區（keep_area）、岩頂照片、這次加進來的。
不加的還有 fill_ban（看過不能用的，例：前面有樹）、岩頂照片分組的 exclude。

清單有 grow 時改成只加不拿掉，補到 grow 張。

用法：atlas_fill.py <atlas64.json> <atlas-ban.json> <相機檔[,…]> <模型點.spz> <路線 crag.json> <照片資料夾> <輸出.json> [輪數]
只算各區岩壁範圍（crag.json 的 walls min／max 外加 PAD 公尺）內的點；相機寬高跟照片對不上的不收；畫面有網格的比例低於 MIN_COVER 的不收（拍到網格外的東西，例：近處沒建到的石頭）。
輸出：每輪加／拿掉哪張、各自補上或損失多少，以及換之前、之後路線周圍缺照片的比例。
"""
import json, math, sys
from pathlib import Path
import cv2, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read
from atlas_frames import zbuf, W, H, F0, W0

STEP, DS, MIN_COS, MAX_D, NEED, NEAR, ROUTE_W = 10, 4, 0.34, 25.0, 3, 0.6, 3.0
SAME_D, SAME_DEG, MIN_GAIN, SIZE, PAD, MIN_COVER = 1.5, 15, 1.3, 0.02, 3.0, 0.5


def seen(P, N, size, c):
    R, C = np.array(c['R'], float), np.array(c['C'], float)
    w0 = c.get('w', W0); h0 = c.get('h', w0 * 16 / 9); f = c.get('f', F0 * w0 / W0)
    k = DS * w0 / W; w, h = int(w0 / k), int(h0 / k); f, cx, cy = f / k, c.get('cx', w0 / 2) / k, c.get('cy', h0 / 2) / k
    d = zbuf(P, size, R, C, w, h, f, cx, cy)
    X = (P - C) @ R.T; z = X[:, 2]; ok = z > 0.4
    u = np.full(len(P), -1); v = np.full(len(P), -1)
    u[ok] = np.rint(X[ok, 0] / z[ok] * f + cx); v[ok] = np.rint(X[ok, 1] / z[ok] * f + cy)
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z < MAX_D)
    i = np.where(ok)[0]; i = i[np.abs(d[v[i], u[i]] - z[i]) < 0.05 * z[i] + 0.15]
    ray = P[i] - C; ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    return i[np.abs((ray * N[i]).sum(1)) > MIN_COS].astype(np.int32), R[2], float(np.isfinite(d).mean())


def main():
    pick_f, ban_f, cam_f, spz, crag, img_dir, out, *rest = sys.argv[1:]
    rounds = int(rest[0]) if rest else 8
    pick = json.loads(Path(pick_f).read_text())
    ban = set(json.loads(Path(ban_f).read_text())) | set(pick.get('user_remove', [])) | set(pick.get('fill_ban', {}))   # fill_ban：看過不能用的（前面有樹…）
    top_ex = Path(__file__).parents[2] / 'data/crag3d/dapaoyan-top-photos.json'
    if top_ex.exists(): ban |= set(json.loads(top_ex.read_text()).get('exclude', []))
    cams = {c['name']: c for f in reversed(cam_f.split(',')) for j in [json.loads(Path(f).read_text())]
            for c in (j['cameras'] if isinstance(j, dict) else j) if 'name' in c and 'C' in c}
    s = read(spz); P = s['xyz'][::STEP].astype(np.float32)
    N = np.load(spz + '.normals.npy')[::STEP].astype(np.float32)
    size = np.full(len(P), SIZE * math.sqrt(STEP), np.float32)
    cg = json.loads(Path(crag).read_text()); routes = cg['routes']
    inside = np.zeros(len(P), bool)   # 只算各區岩壁範圍內的點（範圍外另一塊石頭不算缺口）
    for w in cg['walls']: inside |= np.all((P >= np.array(w['min']) - PAD) & (P <= np.array(w['max']) + PAD), 1)
    seg = np.array([a + (b - a) * t for r in routes for L in [np.array(r['points'])] for a, b in zip(L[:-1], L[1:]) for t in np.linspace(0, 1, 6)])
    near = np.zeros(len(P), bool); near[[i for l in cKDTree(P).query_ball_point(seg, NEAR) for i in l]] = True
    wt = np.where(near, ROUTE_W, 1.0) * inside
    chosen = [n for n in pick['wall'] if n in cams]
    keep = set(pick.get('must', [])) | set(pick.get('user_add', [])) | {n for n, a in zip(pick['wall'], pick['wall_area']) if a in pick.get('keep_area', [])}
    vis, fwd, cover = {}, {}, {}
    def size_ok(n):   # 相機的寬高要跟照片一樣（不一樣的位置不可信）
        im = cv2.imread(str(Path(img_dir) / f'{n}.jpg')); return im is not None and abs(im.shape[1] / im.shape[0] - cams[n].get('w', W0) / cams[n].get('h', 1600)) < 0.01
    names = [n for n in cams if n not in ban and (n in pick['wall'] or size_ok(n))]
    for k, n in enumerate(names):
        vis[n], fwd[n], cover[n] = seen(P, N, size, cams[n])
        if k % 50 == 0: print(f'{k}/{len(names)} 台相機', flush=True)
    good = np.zeros(len(P), np.int16)
    for n in chosen: good[vis[n]] += 1
    top = json.loads(Path(pick_f).read_text())['top']   # 岩頂照片的點另外算進去（用輸出的位置也可，這裡用相機檔有的）
    for n, _ in top:
        if n in vis: good[vis[n]] += 1

    def report(g):
        return {'0 張': round(float((g[near] == 0).mean()), 3), '≤1 張': round(float((g[near] <= 1).mean()), 3)}
    before = report(good); log = []
    added = []
    grow = pick.get('grow')   # 補到 grow 張（只加不拿掉；例：Climbio 的 21 張＋必選，再補到 64 張）
    for _ in range(rounds if not grow else max(0, grow - len(chosen) - len(top))):
        lack = np.maximum(NEED - good, 0) * wt
        def similar(n):
            return any(np.linalg.norm(np.array(cams[n]['C']) - np.array(cams[m]['C'])) < SAME_D and
                       math.degrees(math.acos(np.clip(fwd[n] @ fwd[m], -1, 1))) < SAME_DEG for m in chosen)
        cand = sorted(((float(lack[vis[n]].sum()), n) for n in names if n not in chosen and n not in dict(top) and cover[n] >= MIN_COVER), reverse=True)
        add = next(((g, n) for g, n in cand if not similar(n)), None)
        if not add: break
        g_add, n_add = add
        good[vis[n_add]] += 1
        if grow:
            chosen.append(n_add); added.append(n_add); log.append({'add': n_add, 'gain': round(g_add)})
            print(f'加 {n_add}（補 {g_add:.0f}）', flush=True); continue
        loss = sorted((float(((good[vis[m]] <= NEED) * wt[vis[m]]).sum()), m) for m in chosen if m not in keep and m not in added)
        l_rm, n_rm = loss[0]
        if g_add < MIN_GAIN * l_rm: good[vis[n_add]] -= 1; break
        good[vis[n_rm]] -= 1; chosen.remove(n_rm); chosen.append(n_add); added.append(n_add)
        log.append({'add': n_add, 'gain': round(g_add), 'remove': n_rm, 'loss': round(l_rm)})
        print(f'加 {n_add}（補 {g_add:.0f}）拿掉 {n_rm}（損失 {l_rm:.0f}）', flush=True)
    res = {'before': before, 'after': report(good), 'swaps': log, 'wall': chosen}
    res['cover'] = {n: round(cover[n], 2) for n in chosen if n in cover}
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print('路線周圍缺照片：換之前', before, '換之後', res['after'])


if __name__ == '__main__':
    main()
