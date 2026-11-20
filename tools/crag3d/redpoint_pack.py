"""戶外 3D Topo：輸出紅點貓讀得懂的 Climbio 公開格式 v2（Guidebook 組合）。

- 每面岩壁一張從 3D 模型拍的正面快照（snap.html），是一般的照片畫面，路線投在快照上（圖層），紅點貓照常看 Topo；
  照片的 props.model 記下模型和拍的角度，可以從同一個角度打開模型。
- 3D 模型是 surface: model 的畫面（glTF .glb 或高斯潑濺 .spz），路線的 3D 折線放在疊在它上面的圖層（座標是模型的公尺）。
  整個岩場一個模型，或每面岩壁各一個（一面一面建、品質優先時，一個模型放不下整個岩場的貼圖）。
  給了遮擋表（sight.py 的輸出）：圖層的折線換成它補過點的折線，每條路線的 props.sight 是查表的值，圖層的 props.sight 是規格。
  snaps 資料夾有 overview.json（全區俯視）：那張快照當成一張掛在岩場上的照片，放在最前面（岩場的封面），整個岩場一個模型時 props.model 記拍的鏡頭（紅點貓第一次進 3D 的第一個畫面）。
  岩壁有 sky（現場天空上方、靠近岩壁的顏色）就放進模型畫面的 props.sky，檢視時當背景漸層。
- 路線節點放在組合的 nodes：岩場 → 岩壁 → 路線（難度 yds、頂繩）。
- 合成預覽圖 preview.jpg 與縮圖 thumb.jpg（跟 Climbio Studio 發佈的同一種版面）：組合的 props.preview、照片項目的 props.tile。

用法：redpoint_pack.py <crag_pack 的 crag.json> <snaps 資料夾> <模型 GLB／SPZ，或 岩壁名稱=GLB,岩壁名稱=GLB…> <輸出資料夾> <經度,緯度> [遮擋表 sight.json] [--book=<組合 id>]
--background=<SPZ>：岩壁用網格（GLB）時，四周山林、天空用的高斯潑濺背景：另一個模型畫面，主模型的 props.background 指到它，讀的人畫在模型後面（被模型擋住的地方不畫）。
--book：沿用已經發佈的組合 id（紅點貓靠它認出是同一個岩場的新版本，收過的人才會更新）；沒給就照 crag.json 的 id 算。
"""
import hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np

BOOK = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--book=')), None)
BG = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--background=')), None)
sys.argv = [a for a in sys.argv if not a.startswith(('--book=', '--background='))]
pack_file, snaps_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[4])
models = dict(a.split('=', 1) for a in sys.argv[3].split(',')) if '=' in sys.argv[3] else {None: sys.argv[3]}   # 岩壁名稱 → 模型；None：整個岩場一個
lon, lat = map(float, sys.argv[5].split(','))
sight = json.loads(Path(sys.argv[6]).read_text()) if len(sys.argv) > 6 else None
overview = json.loads((snaps_dir / 'overview.json').read_text()) if (snaps_dir / 'overview.json').exists() else None
pack, snaps = json.loads(pack_file.read_text()), json.loads((snaps_dir / 'snaps.json').read_text())
out.mkdir(parents=True, exist_ok=True)
for old in out.iterdir(): old.unlink()
T, LIC, BY = int(time.time() * 1000), 'CC-BY-SA-4.0', 'Climbio'
A36 = '0123456789abcdefghijklmnopqrstuvwxyz'
fid = lambda key: ''.join(A36[b % 36] for b in hashlib.sha256(f"{pack['id']}:{key}".encode()).digest()[:16])   # 重新輸出時 id 不變
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
nid = lambda key: fid(key)[:12]
base = {'climbio': 2, 't': T, 'license': LIC, 'attribution': BY}


def simplify(a, tol):   # 3D 折線化簡（Douglas–Peucker）
    a = np.asarray(a)
    if len(a) < 3: return a.tolist()
    d = np.linalg.norm(np.cross(a[1:-1] - a[0], a[-1] - a[0]), axis=1) / max(np.linalg.norm(a[-1] - a[0]), 1e-9)
    k = int(d.argmax()) + 1
    return [a[0].tolist(), a[-1].tolist()] if d[k - 1] <= tol else simplify(a[:k + 1], tol)[:-1] + simplify(a[k:], tol)


crag = {'id': nid('crag'), 'type': 'crag', 'name': pack['name'], 'geo': {'type': 'Point', 'coordinates': [lon, lat]}}
walls = {w['name']: {'id': nid('wall:' + w['name']), 'type': 'wall', 'name': w['name'], 'parent': crag['id'], 'props': {'order': i}} for i, w in enumerate(pack['walls'])}
wall_of = lambda num: next((w for w in pack['walls'] if w['routes'][0] <= int(num) <= w['routes'][1]), None)
routes = {}
for r in pack['routes']:
    w = wall_of(r['number'])
    if not w: continue
    routes[r['number']] = {'id': nid('route:' + r['number']), 'type': 'route', 'name': r['name'], 'parent': walls[w['name']]['id'],
                           'props': {'grade': {'yds': r['grade']}, 'style': 'toprope', 'order': int(r['number'])}}
items = []
# 圖檔、模型的檔名帶內容雜湊：網站會快取這些檔，內容換了檔名也要換，收過的人更新時才會下載新的
model_docs, model_of = [], {}
for wname, glb in models.items():
    glb = Path(glb); key = 'model' if wname is None else 'model:' + wname
    spz = glb.suffix == '.spz'
    mv = {**base, 'kind': 'view', 'surface': 'model', 'id': fid(key), 'name': f"{wname or pack['name']} 3D", 'origin': 'self',
          'href': f"model-{sha(glb)[:12]}{glb.suffix}", 'mime': 'model/x-spz' if spz else 'model/gltf-binary', 'sha256': sha(glb),
          'props': {'node': crag['id'] if wname is None else walls[wname]['id'], **({'sky': sky} if (sky := next((w.get('sky') for w in pack['walls'] if w['name'] == wname), None)) else {})}}
    shutil.copyfile(glb, out / mv['href'])
    mine = [r for r in pack['routes'] if r['number'] in routes and (wname is None or wall_of(r['number'])['name'] == wname)]
    seen = lambda r: sight and sight['routes'].get(r['number'])
    ml = {**base, 'kind': 'layer', 'id': fid(key + '-routes'), 'on': mv['id'], 'href': f"{mv['id']}.json", **({'props': {'sight': sight['spec']}} if sight else {}),
          'marks': [{'node': routes[r['number']]['id'], 'geometry': {'type': 'LineString', 'coordinates': seen(r)['points'] if seen(r) else [[round(v, 2) for v in p] for p in simplify(r['points'], 0.04)]},
                     'props': {'label': r['number'], **({'sight': seen(r)['sight']} if seen(r) else {})}} for r in mine]}
    model_docs += [mv, ml]; model_of[wname] = mv['id']
if BG:   # 背景：一個高斯潑濺模型畫面，每個主模型的 props.background 指到它
    bg = Path(BG)
    bv = {**base, 'kind': 'view', 'surface': 'model', 'id': fid('background'), 'name': f"{pack['name']} 背景", 'origin': 'self',
          'href': f"model-{sha(bg)[:12]}{bg.suffix}", 'mime': 'model/x-spz', 'sha256': sha(bg), 'props': {'node': crag['id']}}
    shutil.copyfile(bg, out / bv['href'])
    for d in model_docs:
        if d['kind'] == 'view': d['props']['background'] = bv['id']
    model_docs.append(bv)
if overview:   # 全區俯視的快照：掛在岩場上、沒有路線的照片（岩場的封面）
    vid = fid('snap:overview')
    jpg = f"{vid}-{sha(snaps_dir / overview['file'])[:12]}.jpg"
    shutil.copyfile(snaps_dir / overview['file'], out / jpg)
    view = {**base, 'kind': 'view', 'surface': 'photo', 'id': vid, 'name': pack['name'], 'origin': 'self', 'href': jpg, 'w': overview['w'], 'h': overview['h'],
            'mime': 'image/jpeg', 'sha256': sha(out / jpg), 'props': {'node': crag['id'], **({'model': {'id': model_of[None], 'camera': overview['camera']}} if None in model_of else {})}}   # 紅點貓第一次進 3D 就站在拍這張的地方：卡片放大成這張，再換成 3D
    (out / f"{vid}.json").write_text(json.dumps(view, ensure_ascii=False, separators=(',', ':'))); items.append({'href': f"{vid}.json", 'id': vid})
for s in snaps:
    w = walls[s['wall']]
    vid = fid('snap:' + s['wall'])
    jpg = f"{vid}-{sha(snaps_dir / s['file'])[:12]}.jpg"
    shutil.copyfile(snaps_dir / s['file'], out / jpg)
    view = {**base, 'kind': 'view', 'surface': 'photo', 'id': vid, 'name': s['wall'], 'origin': 'self', 'href': jpg, 'w': s['w'], 'h': s['h'],
            'mime': 'image/jpeg', 'sha256': sha(out / jpg), 'props': {'node': w['id'], 'model': {'id': model_of.get(s['wall'], model_of.get(None)), 'camera': s['camera']}}}
    layer = {**base, 'kind': 'layer', 'id': fid('routes:' + s['wall']), 'on': vid, 'href': f'{vid}.json',
             'marks': [{'node': routes[r['number']]['id'], 'geometry': {'type': 'LineString', 'coordinates': r['line']}, 'props': {'label': r['number']}}
                       for r in s['routes'] if r['number'] in routes and len(r['line']) >= 2]}
    for doc in (view, layer):
        (out / f"{doc['id']}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':'))); items.append({'href': f"{doc['id']}.json", 'id': doc['id']})
for doc in model_docs:   # 模型放在照片後面：讀的人拿第一張照片當入口照片
    (out / f"{doc['id']}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(',', ':'))); items.append({'href': f"{doc['id']}.json", 'id': doc['id']})
# 合成預覽圖（climbio 的 sourcekit/preview-layout.js 的版面）：上面入口照片 720×540，下面每張照片一格 120×90、每列 6 格；縮圖 160×120
from PIL import Image
PW, CH, TW, TH, COLS = 720, 540, 120, 90, 6
photos = [json.loads((out / i['href']).read_text()) for i in items]
photos = [d for d in photos if d['kind'] == 'view' and d['surface'] == 'photo']
fit = lambda im, w, h: im.resize((round(im.width * max(w / im.width, h / im.height)), round(im.height * max(w / im.width, h / im.height)))).crop(((round(im.width * max(w / im.width, h / im.height)) - w) // 2, (round(im.height * max(w / im.width, h / im.height)) - h) // 2, (round(im.width * max(w / im.width, h / im.height)) - w) // 2 + w, (round(im.height * max(w / im.width, h / im.height)) - h) // 2 + h))
sheet = Image.new('RGB', (PW, CH + -(-len(photos) // COLS) * TH), '#e9eff4')
for k, d in enumerate(photos):
    im = Image.open(out / d['href']).convert('RGB')
    if k == 0: sheet.paste(fit(im, PW, CH), (0, 0)); fit(im, 160, 120).save(out / 'thumb.jpg', quality=80)
    sheet.paste(fit(im, TW, TH), ((k % COLS) * TW, CH + k // COLS * TH))
sheet.save(out / 'preview.jpg', quality=80)
tile = {d['id']: k for k, d in enumerate(photos)}
items = [{**i, 'props': {'tile': tile[i['id']]}} if i['id'] in tile else i for i in items]
book = {**base, 'kind': 'set', 'type': 'guidebook', 'id': BOOK or fid('book'), 'name': pack['name'], 'nodes': [crag, *walls.values(), *routes.values()], 'items': items,
        'props': {'preview': {'href': 'preview.jpg', 'w': PW, 'h': sheet.height, 'count': len(photos), 'tile': {'w': TW, 'h': TH, 'cols': COLS, 'y': CH}}}}
(out / 'index.json').write_text(json.dumps(book, ensure_ascii=False, separators=(',', ':')))
size = sum(p.stat().st_size for p in out.iterdir())
print(f"{out}：{len(walls)} 面岩壁、{len(routes)} 條路線、{len(snaps)} 張 Topo、{len(models)} 個 3D 模型，共 {size / 1e6:.2f} MB")
