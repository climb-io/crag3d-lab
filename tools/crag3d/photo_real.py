"""戶外 3D Topo：舊路線照片經過建模時拍的真照片，找出在模型上的拍攝位置（photo_pose.py 的參考圖改用真照片，實驗）。

模型畫出來的參考圖跟舊照片差很多（光線、草、潑濺的糊邊），內角、近拍的舊照片常常配不到；
建模的真照片也是從下往上拍，跟舊照片配得到上千點。sfm_to_model.py 已經把真照片的相機精確對到模型上，所以：
1. 舊照片跟那一區附近、朝著岩壁的真照片用 LoFTR 比，內點最多的 TOP 張當參考圖。
   先全部縮到寬 COARSE 粗比，前 FINE 張再用原本的寬度細比；分數存在 <輸出資料夾>/real-scores.json，重跑不再比。
2. 輸出 views-r<路線>.json（真照片的相機 R、C、焦距、主點，file 是真照片的路徑），
   photo_pose.py batch 用模型在真照片相機的深度把配到的點變成 3D，求舊照片的拍攝位置、投線。

用法：photo_real.py <crag.json> <清單（區 路線編號 舊照片）> <區名清單 walls.txt> <cameras.json（sfm_to_model.py）> <真照片資料夾> <輸出資料夾> <路線編號…>
  輸出 <輸出資料夾>/views-r<編號>.json（同一張底圖的路線共用第一條的；已經有的不再比）與 todo-real.txt。
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from photo_pose import matcher, fit, same_shot, MATCH_W

TOP, RANGE, CONE, COARSE, FINE = 5, 25, 60, 320, 15


def cameras(cam_file, img_dir):
    cams = json.loads(Path(cam_file).read_text())['cameras']
    for c in cams: c['file'] = str(Path(img_dir).resolve() / f"{c['name']}.jpg")
    return [c for c in cams if Path(c['file']).exists()]


def main():
    crag_file, todo_file, walls_file, cam_file, img_dir, out, *numbers = sys.argv[1:]
    walls = {w['name']: w for w in json.loads(Path(crag_file).read_text())['walls']}
    names = Path(walls_file).read_text().split('\n')
    cams, out, run = cameras(cam_file, img_dir), Path(out), matcher()
    cache_file = out / 'real-scores.json'
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    todo, last = [], None   # last：(底圖, 標籤)
    for line in Path(todo_file).read_text().split('\n'):
        if not line.strip(): continue
        i, number, photo_file = line.split()
        if number not in numbers: continue
        photo = cv2.imread(photo_file)
        if last and same_shot(last[0], photo):
            todo.append(f'{last[1]} {number} {photo_file}'); continue
        tag = f'r{number}'
        if (out / f'views-{tag}.json').exists():   # 已經找過的底圖不再比
            todo.append(f'{tag} {number} {photo_file}'); last = (photo, tag); continue
        c = np.array(walls[names[int(i)]]['center'])
        near = []
        for v in cams:
            to = c - np.array(v['C']); d = np.array(v['R'][2], float)
            to[1] = d[1] = 0
            if np.linalg.norm(to) < RANGE and to @ d / np.linalg.norm(to) / np.linalg.norm(d) > np.cos(np.radians(CONE)): near.append(v)
        key = lambda w, v: f'{photo_file}|{v["name"]}|{w}'
        def score(v, w, small):
            if key(w, v) not in cache:
                k0, k1, conf = run(small, fit(cv2.imread(str(v['file'])), w)[0])
                k0, k1 = k0[conf > 0.5], k1[conf > 0.5]
                n = 0
                if len(k0) >= 8:
                    _, mask = cv2.findFundamentalMat(k0, k1, cv2.USAC_MAGSAC, 2.0, 0.999)
                    n = int(mask.sum()) if mask is not None else 0
                cache[key(w, v)] = n
            return cache[key(w, v)]
        small = fit(photo, COARSE)[0]
        coarse = sorted(near, key=lambda v: -score(v, COARSE, small))[:FINE]
        small = fit(photo, MATCH_W)[0]
        scored = sorted(((score(v, MATCH_W, small), v) for v in coarse), key=lambda x: -x[0])
        cache_file.write_text(json.dumps(cache))
        vs = [dict(v, score=n) for n, v in scored[:TOP]]
        (out / f'views-{tag}.json').write_text(json.dumps(vs, ensure_ascii=False))
        todo.append(f'{tag} {number} {photo_file}'); last = (photo, tag)
        print(number, f'{len(near)} 張真照片', [(v['name'], v['score']) for v in vs], flush=True)
    (out / 'todo-real.txt').write_text('\n'.join(todo) + '\n')


if __name__ == '__main__':
    main()
