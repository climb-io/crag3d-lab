"""戶外 3D Topo：輸出手機檢視頁用的資料包（仿岩點貓 3D 牆的 hold/3d/<牆 id>/）。

<輸出>/<岩場 id>/crag.json   分區、岩面、路線（3D 折線）、模型檔名、授權
<輸出>/<岩場 id>/model-<雜湊>.glb   每群一個（gltfpack 壓過、內含 JPEG 貼圖）

模型檔名帶內容雜湊，重新輸出時沒變的模型不用重下載。

模型用修過邊的 wall.edge.glb（edges.py），沒有就用 wall.glb。只放指定的群時，岩面與路線也只留那幾區的。

用法：crag_pack.py <工作資料夾> <routes3d.json> <walls3d.json> <輸出資料夾> <岩場 id> <岩場名稱> [群,群…]
"""
import hashlib, json, shutil, sys
from pathlib import Path

D, routes_file, walls_file, OUT = map(Path, sys.argv[1:5])
crag_id, crag_name = sys.argv[5], sys.argv[6]
only = set(sys.argv[7].split(',')) if len(sys.argv) > 7 else None
dst = OUT / crag_id
dst.mkdir(parents=True, exist_ok=True)
for old in dst.glob('model-*.glb'): old.unlink()

areas = []
for a in json.loads((D / 'groups/areas.json').read_text()):
    groups = [g for g in a['groups'] if only is None or g in only]
    if not groups: continue
    models = []
    for g in groups:
        src = next(p for p in (D / 'groups' / g / 'wall.edge.glb', D / 'groups' / g / 'wall.glb') if p.exists())
        name = f"model-{hashlib.sha256(src.read_bytes()).hexdigest()[:12]}.glb"
        shutil.copyfile(src, dst / name); models.append(name)
    areas.append({'area': a['area'], 'models': models})

kept = {a['area'] for a in areas}
r2 = lambda xs: [[round(v, 2) for v in p] for p in xs]
routes = [{k: r[k] for k in ('number', 'name', 'grade', 'color', 'area')} | {'points': r2(r['points'])}
          for r in json.loads(routes_file.read_text()) if r.get('points') and r['area'] in kept]
walls = [{k: w[k] for k in ('name', 'routes', 'area', 'center', 'top', 'min', 'max', 'standAt')}
         for w in json.loads(walls_file.read_text()) if w.get('center') and w['area'] in kept]
pack = {'format': 'climbio-crag-3d/1', 'id': crag_id, 'name': crag_name, 'areas': areas, 'walls': walls, 'routes': routes,
        'license': '模型與貼圖 CC BY-SA 4.0，署名 Climbio'}
(dst / 'crag.json').write_text(json.dumps(pack, ensure_ascii=False, separators=(',', ':')))
size = sum(p.stat().st_size for p in dst.iterdir())
print(f"{dst}：{len(areas)} 區、{sum(len(a['models']) for a in areas)} 個模型、{len(walls)} 面牆、{len(routes)} 條路線，共 {size / 1e6:.2f} MB")
