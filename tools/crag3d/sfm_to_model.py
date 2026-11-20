"""戶外 3D Topo：把自建模型的 COLMAP 相機精確對到另一個模型（高斯潑濺 .spz）的座標，讓建模真照片能當舊照片對位的橋樑。

各組 wall.json 的相機（at）是靠 GPS 放到地圖上的，差約 1 公尺；潑濺模型是 Atlas 版，座標用 atlas_alignment.json 對到同一張地圖。
1. 粗對：COLMAP 相機中心 → wall.json 的 at，相似變換（Umeyama）。
2. 細對：COLMAP 的點雲（至少 MIN_TRACK 張照片看到的點）對潑濺模型不透明的點做 ICP（含縮放），配對距離從 FAR 逐步縮到 NEAR。
3. 輸出每張照片在潑濺模型座標的相機：R（世界 → 相機，x 右、y 下、z 前）、C、焦距、主點、徑向畸變。

用法（要 pycolmap：runs/dapaoyan-3d/venv）：sfm_to_model.py <COLMAP 重建資料夾> <建模資料夾（含 groups/）> <crop.spz> <輸出 cameras.json>
"""
import json, sys
from pathlib import Path
import numpy as np, pycolmap
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read

MIN_TRACK, FAR, NEAR, SOLID = 3, 1.5, 0.15, 0.5
ROUNDS = int(__import__("os").environ.get("ROUNDS", 30))   # ICP 輪數；0：只用粗對


def similarity(a, b):   # b ≈ s R a + t（Umeyama）
    ma, mb = a.mean(0), b.mean(0); A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (A ** 2).sum() * len(a)
    return s, R, mb - s * R @ ma


def main():
    rec_dir, root, spz, out = sys.argv[1:]
    rec = pycolmap.Reconstruction(rec_dir)
    at = {v['name']: v['at'] for f in Path(root).glob('groups/g*/wall.json') for v in json.loads(f.read_text()).get('views', [])}
    ims = list(rec.images.values())
    names = [im.name[:-4] for im in ims if im.name[:-4] in at]
    a = np.array([rec.images[next(i for i, im in rec.images.items() if im.name[:-4] == n)].projection_center() for n in names])
    b = np.array([at[n] for n in names])
    s, R, t = similarity(a, b)
    print(f'粗對：{len(names)} 張相機，殘差中位數 {np.median(np.linalg.norm(s * a @ R.T + t - b, axis=1)):.2f} m')

    P = np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= MIN_TRACK])
    sp = read(spz); Q = sp['xyz'][sp['alpha'] >= SOLID * 255]
    tree = cKDTree(Q)
    for k in range(ROUNDS):
        far = FAR * (NEAR / FAR) ** (k / (ROUNDS - 1))
        X = s * P @ R.T + t
        d, j = tree.query(X, distance_upper_bound=far)
        ok = np.isfinite(d)
        ds, dR, dt = similarity(X[ok], Q[j[ok]])
        s, R, t = ds * s, dR @ R, ds * dR @ t + dt
        if k % 5 == 0 or k == ROUNDS - 1:
            print(f'ICP {k:2d}：配對距離 {far:.2f} m 內 {ok.mean():.0%} 點，中位數 {np.median(d[ok]):.3f} m')

    cams = []
    for im in ims:
        c = rec.cameras[im.camera_id]; p = im.cam_from_world; p = p() if callable(p) else p
        Rc = p.rotation.matrix() @ R.T
        cams.append({'name': im.name[:-4], 'R': Rc.round(6).tolist(), 'C': (s * R @ im.projection_center() + t).round(4).tolist(),
                     'f': float(c.params[0]), 'cx': float(c.params[1]), 'cy': float(c.params[2]), 'k': float(c.params[3]), 'w': c.width, 'h': c.height})
    Path(out).write_text(json.dumps({'scale': s, 'cameras': cams}))
    print(f'{len(cams)} 張相機 → {out}')


if __name__ == '__main__':
    main()
