"""戶外 3D Topo：照使用者的指正改路線位置（標準答案的一部分，例：「38 太偏左」「55 小煙囪在右邊那個煙囪」）。

修正檔 edits.json：
  { 路線編號: { "shift": 公尺（沿岩壁，正的往右：站在岩壁前面看的右邊）, "top": 公尺（上端再往右移這麼多，從下到上慢慢加）,
                "between": [編號, 編號]（整條移到這兩條路線中間）,
                "ledge": true（往上只到上方的平台：碰到岩面往後退的台階就停）, "trim_top": 公尺（上端截掉這麼多）,
                "no_anchor": true（不照名稱吸到內角、岩角），
                "straight": "top"｜"bottom"｜"mean"（拉成一條直線：整條的左右位置對齊上段、下段或平均，前後照那一半的傾斜度延伸；舊照片是直線、這裡卻 S 形或鋸齒時用），
                "from": 編號（借那條路線的點再左右移：同一面岩壁上，移過去還是貼著岩面），
                "drape_front": true（路線在那一區鏡頭看不到的側面：貼到鏡頭看得到的岩面，不然整條被擋住），
                "why": 使用者怎麼說的或為什麼改 },
    "walls": { 區名: { "yaw": 度（鏡頭繞看的中心轉，正的往左轉：看出去的方向轉向左邊那一面）, "scale": 倍數（鏡頭離看的中心的距離，小於 1 往前）, "why": … } } }
左右移只動沿岩壁的位置（高度、前後不動），之後 sight.py 會貼回岩面、照名稱定位、整理；ledge、trim_top、no_anchor 記在路線的 edit 裡給 sight.py。

用法：route_edits.py <crag.json> <edits.json> <輸出 crag.json>
"""
import json, sys
from pathlib import Path
import numpy as np


def main():
    crag_file, edits_file, out = sys.argv[1:]
    c = json.loads(Path(crag_file).read_text()); edits = json.loads(Path(edits_file).read_text())
    pts = {r['number']: np.array(r['points'], float) for r in c['routes'] if r.get('points')}
    wall = lambda n: next(w for w in c['walls'] if w['routes'][0] <= int(n) <= w['routes'][1])
    for w in c['walls']:
        e = edits.get('walls', {}).get(w['name'])
        if not e: continue
        v = dict(w.get('view') or {'yaw': 0, 'scale': 1}); v['yaw'] = round(v['yaw'] + e.get('yaw', 0), 1); v['scale'] = round(v.get('scale', 1) * e.get('scale', 1), 3); w['view'] = v
        print(f"{w['name']} 鏡頭轉 {e.get('yaw', 0):+}°、距離 ×{e.get('scale', 1)}：{e.get('why', '')}")
    for r in c['routes']:
        e = edits.get(r['number'])
        if not e or e.get('from', r['number']) not in pts: continue   # 沒有座標的路線（例：舊資料漏掉的 20 號）也可以借別條的點
        r['edit'] = {k: e[k] for k in ('ledge', 'trim_top', 'no_anchor', 'drape_front', 'span') if k in e}
        f = np.array(wall(r['number'])['facing'], float); f[1] = 0; f /= np.linalg.norm(f)
        right = np.cross([0, 1, 0], f); right /= np.linalg.norm(right)
        p = pts[e['from']] if 'from' in e else pts[r['number']]; d = e.get('shift', 0.0)   # from：借同一面另一條路線的點（原本描的線太短、或懸在內角的空中）
        if 'between' in e:
            d = np.mean([np.median(pts[k] @ right) for k in e['between']]) - np.median(p @ right)
        y = p[:, 1]; k = (y - y.min()) / max(1e-6, np.ptp(y))
        if 'straight' in e:   # 左右位置對齊上段（最高的 1/4）、下段（最低的 1/4）或平均
            lat = p @ right; ref = {'top': lat[k >= 0.75], 'bottom': lat[k <= 0.25]}.get(e['straight'], lat)
            p = p + (np.median(ref) - lat)[:, None] * right
            dep = p @ f; sel = {'top': k >= 0.5, 'bottom': k <= 0.5}.get(e['straight'], k >= 0)   # 前後也照那一半的傾斜度延伸（另一半跳到前面的岩柱上時拉回來）
            a, b = np.polyfit(y[sel], dep[sel], 1)
            p = p + ((a * y + b) - dep)[:, None] * f
        r['points'] = (p + (d + e.get('top', 0) * k)[:, None] * right).round(3).tolist()
        extra = (f"、上端再 {e['top']:+.2f} m" if 'top' in e else '') + (f"、拉直（對齊{ {'top': '上段', 'bottom': '下段'}.get(e['straight'], '平均') }）" if 'straight' in e else '') + (f" {r['edit']}" if r['edit'] else '')
        print(f"{r['number']} 往{'右' if d > 0 else '左'} {abs(d):.2f} m{extra}：{e.get('why', '')}")
    Path(out).write_text(json.dumps(c, ensure_ascii=False))


if __name__ == '__main__':
    main()
