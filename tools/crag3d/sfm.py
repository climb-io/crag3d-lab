"""戶外 3D Topo 實驗：過濾後的照片做 COLMAP 稀疏重建。不全配對，配對清單以「站」為單位產生（filter.py 分站）；
對不上的照片自然不會被註冊；接不起來的群各自成一個模型。

拍的人站在一個點上下、轉一圈拍，再走到下一個點，所以：
- 同一站的照片全部互相配對（轉一圈，相鄰的一定重疊）。
- 附近的站（站的位置取全站 GPS 中位數，10 m 內）：只配方向接近的照片（羅盤方位差 45°、俯仰差 25° 內），
  朝同一個方向拍的才會重疊；每張在每個鄰站只配最接近的 3 張。
- 岩壁上的站（filter.py 標 top，用來補上半部）另外配它底下那段岩壁下的站：沿岩壁 ±10 m 內的站，平均挑 15 張。
  岩壁上往下拍、岩壁下往上拍，方向不同，不比方位。範圍縮小，配錯的機會少，配對數也少。
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pycolmap

W = Path(sys.argv[1]); OUT = W / 'sfm'; OUT.mkdir(exist_ok=True)
rows = {r['name']: r for r in json.loads((W / 'filter.json').read_text()) if not r['drop']}
keep = sorted(n + '.jpg' for n in rows)
prep = {p['name']: p for p in json.loads((W / 'prep.json').read_text())}
db = OUT / 'colmap.db'
db.unlink(missing_ok=True)
t0 = time.monotonic()
ext = pycolmap.FeatureExtractionOptions(); ext.num_threads = 1; ext.max_image_size = 1600; ext.sift.max_num_features = 4096
# 同一支手機、同一焦距，但直拍橫拍的主點不同，所以每張自己一組內參（PER_IMAGE），由對位估
pycolmap.extract_features(db, W / 'img', image_names=keep, camera_mode=pycolmap.CameraMode.PER_IMAGE, extraction_options=ext, device=pycolmap.Device.cpu)
t1 = time.monotonic()
mo = pycolmap.FeatureMatchingOptions(); mo.num_threads = 1

def xy(n):   # GPS → 公尺（x 東、y 北）
    p = prep[n]; return np.array((p['lng'] * math.cos(math.radians(p['lat'])) * 111320, p['lat'] * 110574)) if p.get('lng') is not None else None


st = {}
for n, r in rows.items(): st.setdefault(r['station'], []).append(n)
ctr = {k: np.median([xy(n) for n in ns if xy(n) is not None], axis=0) for k, ns in st.items() if any(xy(n) is not None for n in ns)}
top = {k: rows[ns[0]]['top'] for k, ns in st.items()}
def gap(a, b):   # 兩張照片的方向差（度）：羅盤方位差＋俯仰差；沒有方位就只比俯仰
    ha, hb = prep[a].get('heading'), prep[b].get('heading')
    dh = 0 if ha is None or hb is None else abs((ha - hb + 180) % 360 - 180)
    dp = abs((prep[a].get('pitch') or 0) - (prep[b].get('pitch') or 0))
    return dh + dp if dh <= 45 and dp <= 25 else None


pairs = set()
add = lambda a, b: a != b and pairs.add(tuple(sorted((a, b))))
for k, ns in st.items():
    for i, a in enumerate(ns):
        for b in ns[i + 1:]: add(a, b)
ks = sorted(ctr)
for i, k in enumerate(ks):
    for j in ks[i + 1:]:
        if top[k] != top[j] or np.linalg.norm(ctr[k] - ctr[j]) > 10: continue
        for a, others in ((a, st[j]) for a in st[k]):
            near = sorted((g, b) for b in others if (g := gap(a, b)) is not None)
            for _, b in near[:3]: add(a, b)
        for b, others in ((b, st[k]) for b in st[j]):
            near = sorted((g, a) for a in others if (g := gap(a, b)) is not None)
            for _, a in near[:3]: add(a, b)
for k in ks:
    if not top[k]: continue
    below = sorted((n for j in ks if not top[j] and abs(ctr[j][0] - ctr[k][0]) <= 10 for n in st[j]), key=lambda n: (rows[n]['station'], n))
    pick = [below[i] for i in np.linspace(0, len(below) - 1, min(15, len(below))).round().astype(int)] if below else []
    for a in st[k]:
        for b in pick: add(a, b)
(OUT / 'pairs.txt').write_text(''.join(f"{a}.jpg {b}.jpg\n" for a, b in sorted(pairs)))
print('配對', len(pairs), '組（', len(st), '站，岩壁上', sum(top.values()), '站）', flush=True)
pycolmap.match_image_pairs(db, matching_options=mo, pairing_options=pycolmap.ImportedPairingOptions(match_list_path=str(OUT / 'pairs.txt')), device=pycolmap.Device.cpu)
t2 = time.monotonic()
opt = pycolmap.IncrementalPipelineOptions(); opt.num_threads = 1; opt.mapper.num_threads = 1; opt.random_seed = 0
opt.min_model_size = 5
models = pycolmap.incremental_mapping(db, W / 'img', OUT, options=opt)
t3 = time.monotonic()
stats = {'input': len(keep), 'seconds': {'features': t1 - t0, 'matching': t2 - t1, 'mapping': t3 - t2}, 'models': []}
for k, m in models.items():
    m.write(OUT / str(k))
    stats['models'].append({'id': k, 'images': m.num_reg_images(), 'points': m.num_points3D(), 'reprojPx': m.compute_mean_reprojection_error(),
                            'names': sorted(i.name for i in m.images.values() if i.has_pose)})
(OUT / 'summary.json').write_text(json.dumps(stats, ensure_ascii=False, indent=1))
print(json.dumps({**stats, 'models': [{k: v for k, v in s.items() if k != 'names'} for s in stats['models']]}, ensure_ascii=False, indent=1))
# 每群照片放進 groups/g<id>/photos（連結），給 build.sh 依序建模
for m in stats['models']:
    d = W.parent / 'groups' / f"g{m['id']}" / 'photos'; d.mkdir(parents=True, exist_ok=True)
    for n in m['names']:
        if not (d / n).exists(): (d / n).symlink_to((W / 'img' / n).resolve())
