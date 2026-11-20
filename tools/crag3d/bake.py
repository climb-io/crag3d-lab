"""戶外 3D Topo：用原始照片重新烘模型的貼圖（Apple 在 8 GB 的 Mac 上貼圖最大 4K，而且會把幾張照片混在一起變糊）。

模型、相機都已放在地圖上（clean.py、align_part.py 的 wall.raw.glb、wall.json）。只換貼圖，不動形狀（模型只來自照片）：
1. 每台相機在縮小的畫面上畫 z-buffer，判斷每個三角形看不看得到（沒被擋住、在畫面內、離天空遮罩 SKYGAP 以上——逆光時岩石邊緣有光暈）：三個角和中心都看得到最好；
   沒有任何照片完整看到的三角形（多半是大三角形跨過遮擋的邊），退一步用中心看得到的照片（分數打 FALLBACK 折）。
   照片在這個三角形上的顏色（扣掉整張照片的曝光差）跟 Apple 貼圖（多張照片的混合，當共識）差超過 AGREE，也算看不到：
   照片前面有模型裡沒有的東西擋住（樹叢、人、背包），只靠深度判斷不出來。
2. 每個三角形選一張照片：分數＝原圖上每公尺幾個像素 × 正對程度，離畫面邊緣太近的扣分；
   再乘上照片在那裡的清晰度（細紋理÷中等紋理的比例；對不到焦、手震時細紋理先消失），近拍但沒對到焦的不會被選上。
   相鄰三角形換照片會有接縫（曝光不同），所以分數不到最好的 SWITCH 倍時，沿用鄰居多數選的照片。
3. 沿用 Apple 的貼圖座標（UV 分塊），放大到 SIZE；每張照片只讀一次，把選它的三角形在貼圖上逐像素投回原圖取色（雙線性）。
4. 兩層合成：每張照片的曝光不同，直接拼會一塊亮一塊暗。大範圍的明暗、顏色（高斯模糊 SIGMA 個貼圖像素以上）用 Apple 的貼圖
   （它本來就是多張照片混合，顏色連續），細節（照片減掉自己的模糊）用選中的那張照片；模糊只在同一張照片烘的範圍裡算，接縫不會出現光暈。
   分塊之間的空白沿用 Apple 原本的貼圖（放大到同樣大小）。
   每張照片顏色都對不上的三角形，也沿用 Apple 的貼圖。
5. 沒有任何照片看得到的三角形刪掉（只看深度）：那是 Apple 在遮住的地方自己搭出來的表面（例如岩塊頂跟後面岩壁之間的岩縫上一整片灰色斜面），
   不是照片拍到的。

焦距：iPhone 12 廣角，COLMAP 各張估計的中位數是長邊的 0.752 倍；主點在畫面中央。

用法：bake.py <wall.raw.glb> <wall.json> <原始照片資料夾> <天空遮罩資料夾> <輸出.raw.glb> [貼圖邊長，預設 8192]
"""
import io, json, sys
from pathlib import Path
import cv2, numpy as np, trimesh
from PIL import Image

FOCAL, VIS, SWITCH, EDGE, SIGMA, FALLBACK, AGREE, SKYGAP = 0.752, 512, 0.8, 0.05, 8, 0.5, 0.12, 0.015
src, views_file, photos, masks, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
SIZE = int(sys.argv[6]) if len(sys.argv) > 6 else 8192
m = trimesh.load(src, force='mesh', process=False)
V, F, UV = m.vertices, m.faces, m.visual.uv
views = json.loads(views_file.read_text())['views']
nf = len(F); ctr = V[F].mean(1); nrm = m.face_normals
print(f'{nf} 面、{len(views)} 台相機、貼圖 {SIZE}')


def camera(v, w, h):   # 地圖 → 畫面像素；Apple 相機座標 x 右、y 上、-z 前
    R, C = np.array(v['rot']), np.array(v['at']); f = FOCAL * max(w, h)
    def proj(P):
        X = (P - C) @ R; z = -X[..., 2]
        return w / 2 + f * X[..., 0] / z, h / 2 - f * X[..., 1] / z, z
    return proj, f, C


# 1、2：每台相機每個三角形的分數
apple = cv2.GaussianBlur(np.asarray(m.visual.material.baseColorTexture.convert('RGB').resize((1024, 1024))).astype(np.float32), (0, 0), 2)
cuv = UV[F].mean(1); apple_col = apple[np.clip(((1 - cuv[:, 1]) * 1024).astype(int), 0, 1023), np.clip((cuv[:, 0] * 1024).astype(int), 0, 1023)]
size = {}
score = np.zeros((len(views), nf), np.float32); part = np.zeros((len(views), nf), np.float32); geo = np.zeros(nf, bool)
for i, v in enumerate(views):
    p = photos / f"{v['name']}.jpeg"
    if not p.exists(): continue
    w, h = Image.open(p).size; size[v['name']] = (w, h)
    k = VIS / max(w, h); W, H = int(w * k), int(h * k)
    proj, f, C = camera(v, W, H)
    x, y, z = proj(V)
    zb = np.full((H, W), np.inf, np.float32)
    good = (z[F] > 0.2).all(1)
    fx, fy = x[F], y[F]
    good &= (fx.max(1) >= 0) & (fx.min(1) < W) & (fy.max(1) >= 0) & (fy.min(1) < H)
    for a, b, c in F[good]:
        xs, ys, zs = x[[a, b, c]], y[[a, b, c]], z[[a, b, c]]
        x0, x1 = int(max(np.floor(xs.min()), 0)), int(min(np.ceil(xs.max()), W - 1)); y0, y1 = int(max(np.floor(ys.min()), 0)), int(min(np.ceil(ys.max()), H - 1))
        if x0 > x1 or y0 > y1: continue
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-12: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d; l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d; l2 = 1 - l0 - l1
        zz = np.where((l0 >= -.02) & (l1 >= -.02) & (l2 >= -.02), l0 * zs[0] + l1 * zs[1] + l2 * zs[2], np.inf)
        sub = zb[y0:y1 + 1, x0:x1 + 1]; np.minimum(sub, zz, out=sub)
    mk = masks / f"{v['name']}.png"
    sky = np.asarray(Image.open(mk).convert('L').resize((W, H))) < 128 if mk.exists() else np.zeros((H, W), bool)
    r = max(int(SKYGAP * max(W, H)), 1); sky = cv2.dilate(sky.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    seen = []
    for P in (ctr, V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]):   # 中心、三個角
        px, py, pz = proj(P)
        xi, yi = np.clip(px.astype(int), 0, W - 1), np.clip(py.astype(int), 0, H - 1)
        vis = good & (pz > 0.2) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & (pz <= zb[yi, xi] * 1.02 + 0.03)
        if not seen: geo |= vis
        seen.append(vis & ~sky[yi, xi])
    ok = np.logical_and.reduce(seen)
    full = Image.open(p).convert('RGB')
    small = cv2.GaussianBlur(np.asarray(full.resize((W, H), Image.BILINEAR)).astype(np.float32), (0, 0), 1.5)
    g = np.asarray(full.convert('L').resize((2 * W, 2 * H), Image.BILINEAR)).astype(np.float32); g1 = cv2.GaussianBlur(g, (0, 0), 1)
    sharp = cv2.GaussianBlur(np.abs(g - g1), (0, 0), 4) / (cv2.GaussianBlur(np.abs(g1 - cv2.GaussianBlur(g, (0, 0), 4)), (0, 0), 4) + 1)
    cx, cy, _ = proj(ctr); col = small[np.clip(cy.astype(int), 0, H - 1), np.clip(cx.astype(int), 0, W - 1)]
    if seen[0].sum() > 20:   # 曝光差：這張照片看得到的三角形上，顏色對 Apple 貼圖的比例（每個色版的中位數）
        gain = np.median(apple_col[seen[0]] + 1, 0) / np.median(col[seen[0]] + 1, 0)
        agree = np.abs(col * gain - apple_col).mean(1) / 255 < AGREE
        ok &= agree; seen[0] &= agree
    D = ctr - C; dist = np.linalg.norm(D, axis=1); cos = np.abs(np.einsum('ij,ij->i', nrm, D)) / dist
    edge = np.minimum.reduce([cx, W - cx, cy, H - cy]) / max(W, H)
    sh = sharp[np.clip((2 * cy).astype(int), 0, 2 * H - 1), np.clip((2 * cx).astype(int), 0, 2 * W - 1)]
    s = f / k / dist * cos * np.clip(edge / EDGE, 0, 1) * sh   # 原圖每公尺像素 × 正對程度 × 清晰度，貼邊扣分
    score[i] = np.where(ok, s, 0); part[i] = np.where(seen[0], s * FALLBACK, 0)
    if i % 20 == 0: print(f'可見度 {i + 1}/{len(views)}', flush=True)

fallback = score.max(0) == 0
score[:, fallback] = part[:, fallback]; del part
best = score.argmax(0); top = score.max(0)
print(f'只有中心看得到的三角形 {(fallback & (top > 0)).mean() * 100:.1f}%')
print(f'沒有任何照片看到的三角形 {(top == 0).mean() * 100:.1f}%')
# 接縫少一點：鄰居多數選的照片分數夠高就沿用（迭代幾次）
adj = m.face_adjacency
for _ in range(3):
    votes = {}
    for a, b in adj:
        votes.setdefault(a, []).append(best[b]); votes.setdefault(b, []).append(best[a])
    changed = 0
    for fi, vs in votes.items():
        cand = max(set(vs), key=vs.count)
        if cand != best[fi] and score[cand, fi] >= SWITCH * top[fi]: best[fi] = cand; changed += 1
    if not changed: break
del score

# 3：逐張照片把選它的三角形烘進貼圖
tex = np.zeros((SIZE, SIZE, 3), np.uint8); src_of = np.full((SIZE, SIZE), -1, np.int16)
T = np.column_stack([UV[:, 0] * SIZE, (1 - UV[:, 1]) * SIZE])   # 貼圖像素座標（上下顛倒）
for i, v in enumerate(views):
    fs = np.nonzero((best == i) & (top > 0))[0]
    if not len(fs): continue
    img = np.asarray(Image.open(photos / f"{v['name']}.jpeg").convert('RGB')).astype(np.float32); h, w = img.shape[:2]
    proj, _, _ = camera(v, w, h)
    for fi in fs:
        a, b, c = F[fi]; xs, ys = T[[a, b, c], 0], T[[a, b, c], 1]
        x0, x1 = int(max(np.floor(xs.min()), 0)), int(min(np.ceil(xs.max()), SIZE - 1)); y0, y1 = int(max(np.floor(ys.min()), 0)), int(min(np.ceil(ys.max()), SIZE - 1))
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-12 or x0 > x1 or y0 > y1: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d; l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d; l2 = 1 - l0 - l1
        inn = (l0 >= -.01) & (l1 >= -.01) & (l2 >= -.01)
        if not inn.any(): continue
        P = l0[inn, None] * V[a] + l1[inn, None] * V[b] + l2[inn, None] * V[c]
        px, py, _ = proj(P)
        px, py = np.clip(px - .5, 0, w - 1.001), np.clip(py - .5, 0, h - 1.001)
        ix, iy = px.astype(int), py.astype(int); dx, dy = (px - ix)[:, None], (py - iy)[:, None]
        col = (img[iy, ix] * (1 - dx) * (1 - dy) + img[iy, ix + 1] * dx * (1 - dy) + img[iy + 1, ix] * (1 - dx) * dy + img[iy + 1, ix + 1] * dx * dy)
        yy, xx = gy[inn].astype(int), gx[inn].astype(int)
        tex[yy, xx] = col.astype(np.uint8); src_of[yy, xx] = i
    del img
    print(f'烘 {v["name"]}：{len(fs)} 面', flush=True)

# 4：兩層合成。模糊用「遮罩內加權」：blur(色×遮罩) ÷ blur(遮罩)，只算遮罩裡的像素
STRIP = 1024   # 一次處理幾列（4 的倍數）：8K 貼圖整張轉成浮點數要 800 MB，分段做


def down(img, mask, y0, y1, x0, x1):   # 縮小 4 倍的 色×遮罩、遮罩（範圍對齊 4 的倍數）
    num = np.zeros(((y1 - y0) // 4, (x1 - x0) // 4, 3), np.float32); den = np.zeros(num.shape[:2], np.float32)
    for r in range(y0, y1, STRIP):
        r1 = min(r + STRIP, y1); mk = mask[r:r1, x0:x1].astype(np.float32); sz = ((x1 - x0) // 4, (r1 - r) // 4)
        num[(r - y0) // 4:(r1 - y0) // 4] = cv2.resize(img[r:r1, x0:x1].astype(np.float32) * mk[..., None], sz, interpolation=cv2.INTER_AREA)
        den[(r - y0) // 4:(r1 - y0) // 4] = cv2.resize(mk, sz, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(num, (0, 0), SIGMA / 4) / np.maximum(cv2.GaussianBlur(den, (0, 0), SIGMA / 4), 1e-3)[..., None]   # 遮罩內加權的模糊


def up(small, sy, sx, y0, y1, x0, x1):   # small 的左上角在原尺寸 (sy, sx)；放大回 [y0:y1, x0:x1]，跟整張一起放大對齊
    a0, a1 = max((y0 - sy) // 4 - 2, 0), min((y1 - sy + 3) // 4 + 2, small.shape[0]); b0, b1 = max((x0 - sx) // 4 - 2, 0), min((x1 - sx + 3) // 4 + 2, small.shape[1])
    big = cv2.resize(small[a0:a1, b0:b1], ((b1 - b0) * 4, (a1 - a0) * 4), interpolation=cv2.INTER_LINEAR)
    oy, ox = y0 - sy - a0 * 4, x0 - sx - b0 * 4
    return big[oy:oy + y1 - y0, ox:ox + x1 - x0]


old = np.asarray(m.visual.material.baseColorTexture.convert('RGB').resize((SIZE, SIZE), Image.BICUBIC))
filled = src_of >= 0
base = down(old, filled, 0, SIZE, 0, SIZE)   # Apple 貼圖在有烘的範圍裡的模糊（縮小 4 倍存）
out_tex = old.copy()
for i in np.unique(src_of[filled]):
    ys, xs = np.nonzero(src_of == i)
    y0, y1 = max(ys.min() - 3 * SIGMA, 0) // 4 * 4, min((ys.max() + 3 * SIGMA + 4) // 4 * 4, SIZE)
    x0, x1 = max(xs.min() - 3 * SIGMA, 0) // 4 * 4, min((xs.max() + 3 * SIGMA + 4) // 4 * 4, SIZE)
    M = src_of == i
    low = down(tex, M, y0, y1, x0, x1)   # 這張照片自己在它範圍裡的模糊
    for r in range(y0, y1, STRIP):
        r1 = min(r + STRIP, y1); mk = M[r:r1, x0:x1]
        if not mk.any(): continue
        v = up(base, 0, 0, r, r1, x0, x1) + tex[r:r1, x0:x1].astype(np.float32) - up(low, y0, x0, r, r1, x0, x1)
        out_tex[r:r1, x0:x1][mk] = np.clip(v, 0, 255).astype(np.uint8)[mk]
del base, M
tex = out_tex
print(f'用原始照片烘的貼圖像素 {filled.mean() * 100:.1f}%（其餘是空白或沒照片，沿用 Apple 的貼圖）')
m.visual = trimesh.visual.TextureVisuals(uv=UV, material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex), metallicFactor=0, roughnessFactor=1))
buf = io.BytesIO(); Image.fromarray(tex).save(buf, 'JPEG', quality=88)
m.visual.material.baseColorTexture = Image.open(io.BytesIO(buf.getvalue()))
print(f'刪掉沒有照片看得到的三角形 {(~geo).sum()} 個、{m.area_faces[~geo].sum():.1f} m²；看得到但顏色都對不上、沿用 Apple 貼圖的 {(geo & (top == 0)).sum()} 個')
m.update_faces(geo); m.remove_unreferenced_vertices()
out.write_bytes(m.export(file_type='glb'))
print(f'{out}：{out.stat().st_size / 1e6:.1f} MB')
