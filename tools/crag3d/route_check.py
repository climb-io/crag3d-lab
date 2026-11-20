"""戶外 3D Topo：紅點貓資料包出貨前，檢查每條路線放大看會不會斷線、有沒有怪線（放大才看得出來的問題）。

- 共用段差一點：連續兩點以上離別條路線的點 NEAR 公尺內、卻不是同一個點（分叉、合併的共用段要座標完全相同，見 sight.py 的 SHARE）。
- 怪線：一步往下超過 BACK 公尺（往回折）、一步沿岩壁前後跳超過 SPIKE 公尺（上段往岩壁裡退 BACK_STEP 以內不算）、轉角超過 FOLD 度（鋸齒、小勾；路線是示意圖，跟 sight.py 的 HOOK 一樣）。
- 被擋（提醒，不算問題：紅點貓把被擋的一段畫成淡色虛線，線不斷）：照紅點貓 3D 的查表（redpoint/model3d.js 的 see）算每個點看不看得到。鏡頭用那一區的鏡頭（整區），
  再放大 1/ZOOM 倍對著路線的下、中、上三處，各再左右轉 TURN 度；看得到的點中間夾著看不到的一段就是斷線。
  看得到的點中間夾著 FILL 點以內看不到的，紅點貓當成看得到（查表 10° 一格，轉角、岩縫附近零星誤判）；整條都看不到、或露出不到 MIN_SEEN 不列（紅點貓整條不畫）。

用法：route_check.py <紅點貓資料包資料夾>（redpoint_pack.py 的輸出）；有問題的路線列出來，結束碼 1（被擋只提醒）。
"""
import base64, json, math, sys, zlib
from pathlib import Path
import numpy as np

BACK, SPIKE, FOLD, ZOOM, TURN, MIN_SEEN, FILL, NEAR, BACK_STEP = 0.15, 0.35, 75, 0.4, 20, 0.3, 3, 0.1, 0.8


def table(b64):
    raw = zlib.decompress(base64.b64decode(b64), -15)
    v = np.frombuffer(raw, np.uint8)
    return np.stack([v & 15, v >> 4], 1).reshape(-1)


def see(pts, tab, spec, cam):
    az_n, (el0, el_step, el_n), dist = spec['az'], spec['el'], spec['dist']
    d = cam - pts; r = np.linalg.norm(d, axis=1); d /= r[:, None]
    a = np.round(np.arctan2(d[:, 0], d[:, 2]) / (2 * np.pi) * az_n).astype(int) % az_n
    e = np.clip(np.round((np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1))) - el0) / el_step).astype(int), 0, el_n - 1)
    v = tab[(np.arange(len(pts)) * el_n + e) * az_n + a]
    vis = (v == 0) | (r <= np.array([0] + dist)[v])
    i = 0
    while i < len(vis):   # 看得到的點中間夾著 FILL 點以內看不到的：當成看得到（紅點貓同一個規則；查表是 10° 一格，轉角、岩縫附近零星誤判）
        if vis[i]: i += 1; continue
        j = i
        while j < len(vis) and not vis[j]: j += 1
        if 0 < i and j < len(vis) and j - i <= FILL: vis[i:j] = True
        i = j
    return vis


def gaps(vis):
    on = np.nonzero(vis)[0]
    if len(on) < 2: return 0
    return int(np.sum(~vis[on[0]:on[-1] + 1]))


def turn(cam, target, deg):
    a = math.radians(deg); v = cam - target
    return target + np.array([v[0] * math.cos(a) + v[2] * math.sin(a), v[1], -v[0] * math.sin(a) + v[2] * math.cos(a)])


def main():
    pack = Path(sys.argv[1])
    docs = [json.loads(f.read_text()) for f in pack.glob('*.json')]
    sets = [d for d in docs if d.get('kind') == 'set']
    nodes = {n['id']: n for s in sets for n in s.get('nodes', [])}
    cams = {d['props']['node']: d['props']['model']['camera'] for d in docs if d.get('kind') == 'view' and (d.get('props') or {}).get('model')}
    layer = next(d for d in docs if d.get('kind') == 'layer' and (d.get('props') or {}).get('sight'))
    spec, bad = layer['props']['sight'], 0
    marks = sorted(layer['marks'], key=lambda m: int(m['props'].get('label') or 0))
    pts = {id(m): np.array(m['geometry']['coordinates'], float) for m in marks}
    for m in marks:
        p = pts[id(m)]; label = m['props'].get('label')
        c = cams.get(nodes[m['node']].get('parent'))
        issues = []
        for o in marks:
            if o is m: continue
            q = pts[id(o)]
            d = np.linalg.norm(p[:, None] - q[None], axis=2).min(1)
            close = (d < NEAR) & (d > 1e-6)
            if np.any(close[1:] & close[:-1]): issues.append(f'跟 {o["props"].get("label")} 號的共用段差一點')
        for i in range(1, len(p)):
            if p[i, 1] - p[i - 1, 1] < -BACK: issues.append(f'往下 {p[i, 1] - p[i - 1, 1]:.2f} m（第 {i} 點）')
        if c:
            f = np.array(c['target']) - np.array(c['position']); f[1] = 0; f /= np.linalg.norm(f)
            dd = np.diff(p @ f); top = p[:, 1].min() + 0.7 * np.ptp(p[:, 1])
            for i in np.nonzero(np.abs(dd) > SPIKE)[0]:
                if dd[i] > 0 and dd[i] <= BACK_STEP and p[i + 1, 1] > top: continue   # 上段往岩壁裡退（台階、岩頂往後收，sight.py 的 BACK_STEP）是岩壁本來的樣子
                issues.append(f'前後跳 {abs(dd[i]):.2f} m（第 {i + 1} 點）')
        for i in range(1, len(p) - 1):
            u, v = p[i] - p[i - 1], p[i + 1] - p[i]
            if np.linalg.norm(u) > 0.05 and np.linalg.norm(v) > 0.05:
                ang = math.degrees(math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
                if ang > FOLD: issues.append(f'折 {ang:.0f}°（第 {i} 點）')
        if c and m['props'].get('sight'):
            tab, cam0, tgt0 = table(m['props']['sight']), np.array(c['position']), np.array(c['target'])
            views = [('整區', cam0)]
            for name, k in (('下', 0.15), ('中', 0.5), ('上', 0.85)):
                t = p[int(k * (len(p) - 1))]
                for deg in (0, -TURN, TURN):
                    views.append((f'放大{name}{deg:+d}°' if deg else f'放大{name}', turn(t + (cam0 - tgt0) * ZOOM, t, deg)))
            hidden = [f'{name} {gaps(vis)} 點' for name, cam in views if (vis := see(p, tab, spec, cam)).mean() >= MIN_SEEN and gaps(vis)]
        else: hidden = []
        print(f'{label:>3}  {len(p):>2} 點  ' + ('、'.join(issues) if issues else 'ok') + ('（被擋、畫虛線：' + '、'.join(hidden) + '）' if hidden else ''))
        bad += bool(issues)
    print(f'{bad} 條有問題')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
