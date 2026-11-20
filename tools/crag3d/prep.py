"""戶外 3D Topo 實驗第一步：縮圖並算每張照片的品質指標（不判定，判定在 filter.py）。"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageOps, ExifTags

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
LONG = 1600
(OUT / 'img').mkdir(parents=True, exist_ok=True)
GPS = {v: k for k, v in ExifTags.GPSTAGS.items()}


def deg(v, ref):
    d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    return -d if ref in ('S', 'W') else d


def meta(im):
    ex = im.getexif(); g = ex.get_ifd(0x8825); e = ex.get_ifd(0x8769)
    m = {'time': e.get(0x9003), 'focal35': e.get(0xA405)}
    if g.get(GPS['GPSLatitude']):
        m['lat'] = deg(g[GPS['GPSLatitude']], g[GPS['GPSLatitudeRef']])
        m['lng'] = deg(g[GPS['GPSLongitude']], g[GPS['GPSLongitudeRef']])
        m['alt'] = float(g.get(GPS['GPSAltitude'], 0))
        m['accuracy'] = float(g[GPS['GPSHPositioningError']]) if GPS['GPSHPositioningError'] in g else None
        m['heading'] = float(g[GPS['GPSImgDirection']]) if GPS['GPSImgDirection'] in g else None
    return m


def metrics(bgr):
    small = cv2.resize(bgr, None, fx=800 / max(bgr.shape[:2]), fy=800 / max(bgr.shape[:2]), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1] / 255, hsv[..., 2] / 255
    sky = ((v > 0.75) & (s < 0.12)) | ((h >= 95) & (h <= 130) & (s > 0.2) & (v > 0.5))
    green = (h >= 30) & (h <= 90) & (s > 0.25) & (v > 0.15)
    return {
        'sharp': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'bright': float(gray.mean()), 'dark': float((gray < 20).mean()), 'blown': float((gray > 245).mean()),
        'sky': float(sky.mean()), 'green': float(green.mean()),
        'dhash': ''.join('1' if b else '0' for b in (lambda a: (a[:, 1:] > a[:, :-1]).flatten())(cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA))),
    }


rows = []
for p in sorted(SRC.glob('IMG_*.jpeg')):
    im = Image.open(p); m = meta(im)
    im = ImageOps.exif_transpose(im); im.thumbnail((LONG, LONG), Image.LANCZOS)
    ex = Image.open(p).getexif(); ex[0x0112] = 1
    out = OUT / 'img' / (p.stem + '.jpg'); im.save(out, quality=90, exif=ex.tobytes())
    bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    rows.append({'name': p.stem, 'w': im.width, 'h': im.height, **m, **metrics(bgr)})
    print(p.stem, flush=True)
(OUT / 'prep.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
