"""整片岩場拼成一個 GLB：各面岩壁的模型（都對齊在同一個地圖座標，assemble.py --clean 直接刪過岩壁前的植物、地面雜物和碎塊）
沿岩壁方向切成一段一段，每段只用一面岩壁的模型，不讓兩個模型疊在同一個地方。

- 岩壁方向：全部路線點水平分布的主軸。相鄰兩面以兩邊路線範圍的中點分界，分界兩側各多留 OVERLAP 公尺蓋住對齊誤差；
  兩端的岩壁往外不切。
- 貼圖：裁到這一段用到的範圍，長邊縮到 TEX 以內（整片一起載入，手機記憶體才夠）。
- 一次只處理一面：切好就輸出成 GLB 位元組、放掉貼圖，最後在 GLB 的層次合併（不再解碼貼圖），8 GB 的 Mac 才跑得動。

用法：join.py <crag.json（全部岩壁與路線）> <輸出.glb> <gltfpack> <岩壁名稱=模型.raw.glb> …
"""
import io, json, struct, subprocess, sys
from pathlib import Path
import numpy as np, trimesh
from PIL import Image

OVERLAP, TEX, MARGIN = 0.1, 4096, 4
crag = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2]); gltfpack = sys.argv[3]
parts = dict(a.split('=', 1) for a in sys.argv[4:])
walls = [w for w in crag['walls'] if w['name'] in parts]
route_pts = lambda w: np.array([p for r in crag['routes'] if w['routes'][0] <= int(r['number']) <= w['routes'][1] for p in r['points']])
allp = np.concatenate([route_pts(w) for w in walls])[:, [0, 2]]
c0 = allp.mean(0); along = np.linalg.svd(allp - c0, full_matrices=False)[2][0]
pos = lambda xz: (xz - c0) @ along
span = {w['name']: (pos(route_pts(w)[:, [0, 2]]).min(), pos(route_pts(w)[:, [0, 2]]).max()) for w in walls}
walls.sort(key=lambda w: sum(span[w['name']]))
cuts = [(span[a['name']][1] + span[b['name']][0]) / 2 for a, b in zip(walls, walls[1:])]
bounds = [-np.inf] + cuts + [np.inf]

glbs = []
for i, w in enumerate(walls):
    lo, hi = bounds[i] - OVERLAP, bounds[i + 1] + OVERLAP
    m = trimesh.load(parts[w['name']], force='mesh', process=False)
    s = pos(m.triangles_center[:, [0, 2]]); faces = m.faces[(s >= lo) & (s <= hi)]
    used = np.unique(faces); remap = np.full(len(m.vertices), -1); remap[used] = np.arange(len(used))
    tex = m.visual.material.baseColorTexture.convert('RGB'); W, H = tex.size
    uv = m.visual.uv[used]; px = uv[:, 0] * W; py = (1 - uv[:, 1]) * H   # trimesh 的 v 朝上，圖片的列朝下
    x0, y0 = max(int(px.min()) - MARGIN, 0), max(int(py.min()) - MARGIN, 0)
    x1, y1 = min(int(np.ceil(px.max())) + MARGIN, W), min(int(np.ceil(py.max())) + MARGIN, H)
    crop = tex.crop((x0, y0, x1, y1)); cw, ch = crop.size
    if max(cw, ch) > TEX: crop = crop.resize((round(cw * TEX / max(cw, ch)), round(ch * TEX / max(cw, ch))), Image.LANCZOS)
    buf = io.BytesIO(); crop.save(buf, 'JPEG', quality=85)
    part = trimesh.Trimesh(m.vertices[used], remap[faces], process=False)
    part.visual = trimesh.visual.TextureVisuals(uv=np.column_stack([(px - x0) / cw, 1 - (py - y0) / ch]),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.open(io.BytesIO(buf.getvalue())), metallicFactor=0, roughnessFactor=1))
    sc = trimesh.Scene(); sc.add_geometry(part, geom_name=w['name']); glbs.append(sc.export(file_type='glb'))
    print(f"{w['name']}：沿岩壁 {lo:.1f}～{hi:.1f} m，{len(m.faces)} 面 → {len(faces)}，貼圖 {W}×{H} 裁成 {cw}×{ch} → {crop.size[0]}×{crop.size[1]}")
    del m, tex, crop, part, sc


def merge(glbs):
    """幾個 GLB 合成一個：各自的 JSON 陣列接起來、索引加上前面的數量，二進位區塊依序接（4 位元組對齊）。"""
    J = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': []}]}; B = bytearray()
    for g in glbs:
        n = struct.unpack('<I', g[12:16])[0]; js = json.loads(g[20:20 + n]); bn = g[20 + n + 8:]
        off = {k: len(J.get(k, [])) for k in ('accessors', 'bufferViews', 'images', 'textures', 'materials', 'meshes', 'nodes', 'samplers')}
        base = len(B); B += bn; B += b'\0' * (-len(B) % 4)
        for v in js.get('bufferViews', []): v['buffer'] = 0; v['byteOffset'] = v.get('byteOffset', 0) + base
        for a in js.get('accessors', []): a['bufferView'] += off['bufferViews']
        for im in js.get('images', []): im['bufferView'] += off['bufferViews']
        for t in js.get('textures', []):
            t['source'] += off['images']
            if 'sampler' in t: t['sampler'] += off['samplers']
        for mt in js.get('materials', []):
            for tx in [mt.get('pbrMetallicRoughness', {}).get('baseColorTexture')]:
                if tx: tx['index'] += off['textures']
        for me in js.get('meshes', []):
            for pr in me['primitives']:
                pr['attributes'] = {k: v + off['accessors'] for k, v in pr['attributes'].items()}
                if 'indices' in pr: pr['indices'] += off['accessors']
                if 'material' in pr: pr['material'] += off['materials']
        for nd in js.get('nodes', []):
            if 'mesh' in nd: nd['mesh'] += off['meshes']
            if 'children' in nd: nd['children'] = [c + off['nodes'] for c in nd['children']]
        J['scenes'][0]['nodes'] += [i + off['nodes'] for i in js['scenes'][js.get('scene', 0)]['nodes']]
        for k in off: J.setdefault(k, []).extend(js.get(k, []))
    J = {k: v for k, v in J.items() if v != []}; J['buffers'] = [{'byteLength': len(B)}]
    js = json.dumps(J, ensure_ascii=False).encode(); js += b' ' * (-len(js) % 4)
    return b''.join([struct.pack('<III', 0x46546C67, 2, 28 + len(js) + len(B)), struct.pack('<II', len(js), 0x4E4F534A), js, struct.pack('<II', len(B), 0x004E4942), bytes(B)])


raw = out.with_suffix('.raw.glb'); raw.write_bytes(merge(glbs))
subprocess.run([gltfpack, '-i', str(raw), '-o', str(out), '-kn'], check=True, capture_output=True)
print(f'{out}：{out.stat().st_size / 1e6:.1f} MB')
