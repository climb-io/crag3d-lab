"""戶外 3D Topo：量模型的破洞（只量、不補：模型只能來自照片）。
從每面岩壁打開時的角度（snaps.json 的相機）和 3D 頁轉得到的角度（左右各轉 TURN、上下傾斜 TILT），
用 numpy 逐三角形的 z-buffer 畫模型（150×200，不用光線測試，記憶體只要幾百 MB），量兩種破洞：
- 洞：岩壁範圍裡沒有表面的像素（被模型圍住、或窄於 BRIDGE 格的缺口），除以岩壁範圍。
- 腳下：岩壁正下方沒有表面的像素（同一欄上面有模型），除以岩壁範圍。
另外印出貼圖清晰度（每公尺岩面幾個貼圖像素，貼圖全部攤在模型表面上算）。

用法：holes.py <snaps.json> <模型 wall.raw.glb 或 wall.edge.raw.glb> [模型 2 …]；HOLES_DUMP＝資料夾，存每個角度的圖（灰底是沒有表面）
"""
import json, os, struct, sys
from pathlib import Path
import cv2, numpy as np, trimesh
from scipy import ndimage

W, H, BRIDGE = 150, 200, 12
TURN, TILT = 0.45, 0.25
snaps, parts = json.loads(Path(sys.argv[1]).read_text()), sys.argv[2:]
DUMP = os.environ.get('HOLES_DUMP')


def keep_values(path, n):   # 頂點留值（COLOR_0 的 alpha）；沒有就全留
    b = Path(path).read_bytes(); k = struct.unpack('<I', b[12:16])[0]; js = json.loads(b[20:20 + k]); binc = b[20 + k + 8:]
    a = js['meshes'][0]['primitives'][0]['attributes'].get('COLOR_0')
    if a is None: return np.ones(n)
    acc = js['accessors'][a]; bv = js['bufferViews'][acc['bufferView']]; s = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    return np.frombuffer(binc[s:s + acc['count'] * 16], np.float32).reshape(-1, 4)[:, 3].copy()


# 所有塊合成一個網格：每個面記得顏色怎麼取（貼圖＋uv，或頂點顏色）
V, F, keep, uv, tex_of, texs, vcol, area = [], [], [], [], [], [], [], 0.0
for i, p in enumerate(parts):
    m = trimesh.load(p, force='mesh', process=False)
    off = sum(len(v) for v in V)
    V.append(m.vertices); F.append(m.faces + off); keep.append(keep_values(p, len(m.vertices))); area += m.area
    uv.append(m.visual.uv); vcol.append(np.zeros((len(m.vertices), 3)))
    texs.append(np.asarray(m.visual.material.baseColorTexture.convert('RGB'))); tex_of.append(np.full(len(m.faces), i))
VV, FF = np.vstack(V), np.vstack(F)
keep, uv, tex_of, vcol = np.concatenate(keep), np.vstack(uv), np.concatenate(tex_of), np.vstack(vcol)
fkeep = (keep[FF] >= 0.5).any(1)   # 整個面都在切線外的不畫（跟檢視頁的 alphaTest 差不多）


def zbuffer(cam, verts, faces, color_of):
    """從相機看：逐三角形畫進 H×W 的深度圖（不用光線測試，記憶體只要幾 MB）。color_of(面 index, 重心座標) 回 RGB"""
    pos, tgt = np.array(cam['position']), np.array(cam['target'])
    fwd = (tgt - pos) / np.linalg.norm(tgt - pos); right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    t = np.tan(np.radians(cam['fov']) / 2)
    rel = verts - pos; z = rel @ fwd
    sx = ((rel @ right) / np.maximum(z, 1e-6) / (t * W / H) + 1) / 2 * W - 0.5
    sy = (1 - (rel @ up) / np.maximum(z, 1e-6) / t) / 2 * H - 0.5
    depth, col = np.full((H, W), np.inf), np.zeros((H, W, 3))
    for fi in np.flatnonzero((z[faces] > 0.3).all(1)):
        i = faces[fi]; x, y = sx[i], sy[i]
        x0, x1, y0, y1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())), W - 1), max(int(np.floor(y.min())), 0), min(int(np.ceil(y.max())), H - 1)
        if x0 > x1 or y0 > y1: continue
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(den) < 1e-9: continue
        a = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
        b = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
        c = 1 - a - b
        ins = (a >= 0) & (b >= 0) & (c >= 0)
        if not ins.any(): continue
        zz = 1 / (a / z[i[0]] + b / z[i[1]] + c / z[i[2]])   # 透視正確的深度
        win = ins & (zz < depth[y0:y1 + 1, x0:x1 + 1])
        if not win.any(): continue
        depth[y0:y1 + 1, x0:x1 + 1][win] = zz[win]
        col[y0:y1 + 1, x0:x1 + 1][win] = color_of(fi, np.stack([a[win], b[win], c[win]], 1))
    ys, xs = np.mgrid[0:H, 0:W]
    d = fwd + ((xs + 0.5) / W * 2 - 1)[..., None] * t * W / H * right + (1 - (ys + 0.5) / H * 2)[..., None] * t * up
    return pos, d, depth, col   # depth 是沿著 fwd 的距離；點 = pos + d * depth


def model_color(fi, bary):
    tx = texs[tex_of[fi]]; u = bary @ uv[FF[fi]]
    return tx[((1 - u[:, 1]) % 1 * (tx.shape[0] - 1)).astype(int), (u[:, 0] % 1 * (tx.shape[1] - 1)).astype(int)]


def render(cam):
    """從相機畫模型：沒畫到的像素深度是 nan"""
    fmap = np.flatnonzero(fkeep)
    pos, d, depth, col = zbuffer(cam, VV, FF[fkeep], lambda fi, b: model_color(fmap[fi], b))
    depth[np.isinf(depth)] = np.nan
    return pos, d, depth, col


def holes(dist):
    """岩壁範圍（模型圍起來的地方，窄縫也算）裡沒打到的像素"""
    hit = ~np.isnan(dist)
    area = ndimage.binary_fill_holes(ndimage.binary_closing(hit, iterations=BRIDGE, border_value=0)) | hit
    return area & ~hit, area


def views_of(cam):
    """正面，加上 3D 頁轉得到的角度：左右各轉 TURN、往上往下傾斜 TILT（弧度）"""
    pos, tgt = np.array(cam['position']), np.array(cam['target'])
    off = pos - tgt; r = np.linalg.norm(off); az = np.arctan2(off[0], off[2]); po = np.arccos(off[1] / r)
    out = []
    for da, dp in ((0, 0), (-TURN, 0), (TURN, 0), (0, -TILT), (0, TILT)):
        a, p_ = az + da, po + dp
        out.append({**cam, 'position': (tgt + r * np.array([np.sin(p_) * np.sin(a), np.cos(p_), np.sin(p_) * np.cos(a)])).tolist()})
    return out


tex_px = sum(t.shape[0] * t.shape[1] for t in texs)
print(f"模型 {len(FF)} 面、表面積 {area:.0f} m²、貼圖 {'＋'.join(f'{t.shape[1]}×{t.shape[0]}' for t in texs)}：每公尺 {np.sqrt(tex_px / area):.0f} 個貼圖像素")
for s in snaps:
    for tag, cam in zip(['正面', '左轉', '右轉', '往上', '往下'], views_of(s['camera'])):
        pos, d, dist, col = render(cam)
        hole, area_px = holes(dist)
        hit = ~np.isnan(dist)
        foot = ~hit & np.maximum.accumulate(hit, axis=0) & ~area_px   # 岩壁正下方：同一欄上面有模型
        print(f"{s['wall']} {tag}：洞 {hole.sum() / max(area_px.sum(), 1):.1%}、腳下沒有表面 {foot.sum() / max(area_px.sum(), 1):.1%}", flush=True)
        if DUMP:
            img = np.where(hit[..., None], col, 200).astype(np.uint8)
            cv2.imwrite(str(Path(DUMP) / f"{s['wall']}-{tag}.png"), cv2.resize(img[..., ::-1], (W * 3, H * 3), interpolation=cv2.INTER_NEAREST))
