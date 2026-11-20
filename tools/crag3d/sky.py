"""戶外 3D Topo：每張照片的天空遮罩（給 capture_lite 的 masks=，白色是要建模的部分、黑色是天空）。

天空：亮、平滑（岩石、草木都有紋理，天空沒有）、不是鮮豔的顏色（藍天除外），而且跟照片上緣連在一起（白色的岩石不會被當成天空），
再加上跟天空連在一起、很亮又沒什麼顏色的雲；
往外多蓋一點，岩頂邊緣的天空光暈也遮掉。在長邊 800 的縮圖上算，再放大回原圖。照片照 EXIF 轉正，遮罩跟轉正後的照片一樣大。

用法：sky.py <照片資料夾> <遮罩輸出資料夾> [抽查圖.jpg]
"""
import sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageOps

SIDE, WIN, SMOOTH, GROW = 800, 9, 0.022, 5   # 縮圖長邊、紋理窗口、平滑門檻（灰階標準差）、往外多蓋（縮圖像素）


def sky_mask(rgb):
    k = SIDE / max(rgb.shape[:2]); small = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV); h, s, v = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    g = small.astype(np.float32).mean(2) / 255
    std = np.sqrt(np.maximum(cv2.blur(g * g, (WIN, WIN)) - cv2.blur(g, (WIN, WIN)) ** 2, 0))
    sky = (std < SMOOTH) & (v > 0.5) & ((s < 0.25) | ((h >= 95) & (h <= 130)))
    sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=4)
    top = [i for i in np.unique(lab[0][sky[0] > 0]) if i and stats[i, cv2.CC_STAT_AREA] > 200]
    sky = np.isin(lab, top)
    cloud = sky | ((v > 0.8) & (s < 0.2))   # 有紋理的雲：很亮、沒什麼顏色，跟天空連在一起（太陽照到的岩石約 0.7，不能再低）
    n, lab = cv2.connectedComponents(cloud.astype(np.uint8), connectivity=4)
    sky = np.isin(lab, np.unique(lab[sky])).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - sky, connectivity=4)   # 被天空整圈包住的（雲裡較暗的部分）也算
    hw = sky.shape
    inner = [i for i in range(1, n) if stats[i, 0] > 0 and stats[i, 1] > 0 and stats[i, 0] + stats[i, 2] < hw[1] and stats[i, 1] + stats[i, 3] < hw[0]]
    sky[np.isin(lab, inner)] = 1
    sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))   # 雲裡的細紋補起來
    sky = cv2.dilate(sky, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1, 2 * GROW + 1)))
    return cv2.resize(sky, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
sheet, fracs = [], []
for f in sorted(src.iterdir()):
    if f.suffix.lower() not in ('.jpg', '.jpeg', '.heic', '.png'): continue
    rgb = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert('RGB'))
    sky = sky_mask(rgb); fracs.append(sky.mean())
    Image.fromarray(np.where(sky, 0, 255).astype(np.uint8)).save(out / f'{f.stem}.png')
    if len(sys.argv) > 3 and sky.mean() > 0.05:
        t = rgb.copy(); t[sky] = (t[sky] * 0.3 + np.array([255, 0, 255]) * 0.7).astype(np.uint8)
        sheet.append(cv2.resize(t, (240, int(240 * t.shape[0] / t.shape[1]))))
fracs = np.array(fracs)
print(f'{len(fracs)} 張：有天空的 {np.mean(fracs > 0.01) * 100:.0f}%，天空佔畫面中位 {np.median(fracs[fracs > 0.01]) * 100:.0f}%')
if sheet:
    sheet = sheet[::max(1, len(sheet) // 24)][:24]; hmax = max(s.shape[0] for s in sheet)
    rows = [np.hstack([np.pad(s, ((0, hmax - s.shape[0]), (0, 0), (0, 0)), constant_values=255) for s in sheet[i:i + 8]]) for i in range(0, len(sheet), 8)]
    w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    Image.fromarray(np.vstack(rows)).save(sys.argv[3], quality=80)
