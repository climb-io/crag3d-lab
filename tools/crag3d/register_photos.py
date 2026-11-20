"""戶外 3D Topo：建模時沒對到位置的照片（例：岩壁上方往下拍的），對到網格模型的座標（實驗）。

跟舊路線照片同一套（photo_pose.py 的 pose）：照片先跟 GPS 附近 RANGE 公尺內、有位置的真照片用 LoFTR 粗篩（寬 COARSE）、
前 FINE 張細比，最像的幾張當參考圖；參考圖上配到的點用模型深度換成 3D，PnP 算這張照片的拍攝位置（焦距用建模時的 1103 px）。

用法：register_photos.py <模型點.spz> <cameras.json（真照片相機）> <prep.json（GPS）> <照片資料夾> <輸出.json> <照片名…>
"""
import json, math, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read
from photo_pose import matcher, fit, pose, MATCH_W, SOLID

RANGE, COARSE, FINE, TOP = 15, 320, 12, 5


def main():
    spz, cam_file, prep_file, img_dir, out, *names = sys.argv[1:]
    s = read(spz); Q = s['xyz'][s['alpha'] >= SOLID * 255]
    cams = {c['name']: c for c in json.loads(Path(cam_file).read_text())['cameras']}
    prep = {p['name']: p for p in json.loads(Path(prep_file).read_text())}
    gd = lambda a, b: math.hypot((a['lat'] - b['lat']) * 111320, (a['lng'] - b['lng']) * 111320 * math.cos(math.radians(a['lat'])))
    run = matcher()
    res = json.loads(Path(out).read_text()) if Path(out).exists() else {}   # 保留之前對過的照片
    for n in names:
        photo = cv2.imread(str(Path(img_dir) / f'{n}.jpg')); p = prep[n]
        near = [c for m, c in cams.items() if m != n and prep.get(m, {}).get('lat') and gd(p, prep[m]) < RANGE]
        def score(c, w):
            a = fit(photo, w)[0]; b = fit(cv2.imread(str(Path(img_dir) / f"{c['name']}.jpg")), w)[0]
            k0, k1, conf = run(a, b); k0, k1 = k0[conf > 0.5], k1[conf > 0.5]
            if len(k0) < 8: return 0
            _, m = cv2.findFundamentalMat(k0, k1, cv2.USAC_MAGSAC, 2.0, 0.999)
            return int(m.sum()) if m is not None else 0
        coarse = sorted(near, key=lambda c: -score(c, COARSE))[:FINE]
        best = sorted(((score(c, MATCH_W), c) for c in coarse), key=lambda x: -x[0])[:TOP]
        refs = [(dict(c, file=f"{c['name']}.jpg"), cv2.imread(str(Path(img_dir) / f"{c['name']}.jpg"))) for _, c in best]
        r = pose(Q, photo, refs, run)
        res[n] = {'refs': [(c['name'], k) for k, c in best], 'near': len(near)}
        if r: res[n].update(C=r['C'].round(3).tolist(), R=r['R'].round(6).tolist(), fov=r['fov'], inliers=r['inliers'], matches=r['matches'], ratio=round(r['inliers'] / max(1, r['matches']), 2))
        print(n, f"附近 {len(near)} 張", '最像', res[n]['refs'][:3], '內點', res[n].get('inliers'), '/', res[n].get('matches'), flush=True)
        Path(out).write_text(json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
