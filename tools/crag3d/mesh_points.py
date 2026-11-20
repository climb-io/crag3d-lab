"""戶外 3D Topo：網格模型（.glb，照片貼圖）在表面取點，存成跟高斯潑濺一樣格式的點檔（.spz），給作者端計算用（不發佈）。

岩壁改用網格（潑濺只用少數照片訓練，岩面失真）後，遮擋表（sight.py）、貼面與延伸、舊照片投線（photo_pose.py）、裁切都還是讀「模型的點」：
每平方公尺取 DENSITY 個點（表面均勻），顏色讀貼圖（往上延伸靠顏色認出草叢），不透明、每個點 SIZE 公尺大小。

法線另存 <輸出>.normals.npy（每個點所在三角形的法線，挖背景 carve_bg.py 用）。

用法（要 trimesh：runs/dapaoyan-3d/venv）：mesh_points.py <模型.glb> <輸出.spz>
"""
import sys
from pathlib import Path
import numpy as np, trimesh
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import write

DENSITY, SIZE, FRAC = 2500, 0.02, 9


def main():
    src, out = sys.argv[1:]
    scene = trimesh.load(src)
    meshes = scene.dump() if hasattr(scene, 'dump') else [scene]
    xyz, rgb, nrm = [], [], []
    for m in meshes:
        n = max(1, int(m.area * DENSITY))
        p, fi = trimesh.sample.sample_surface(m, n)
        mat = getattr(m.visual, 'material', None); img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
        if img is not None and getattr(m.visual, 'uv', None) is not None:   # 貼圖顏色：取樣點在三角形裡的重心座標內插 UV，讀貼圖
            bary = trimesh.triangles.points_to_barycentric(m.triangles[fi], p)
            uv = (m.visual.uv[m.faces[fi]] * bary[:, :, None]).sum(1)
            im = np.asarray(img.convert('RGB')); h, w = im.shape[:2]
            c = im[np.clip(((1 - uv[:, 1] % 1) * (h - 1)).round().astype(int), 0, h - 1), np.clip(((uv[:, 0] % 1) * (w - 1)).round().astype(int), 0, w - 1)] / 255.0
        else: c = np.full((len(p), 3), 0.5)
        xyz.append(p); rgb.append(c); nrm.append(m.face_normals[fi])
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    np.save(out + '.normals.npy', np.concatenate(nrm).astype(np.float16))
    color = np.clip(np.round(((rgb - 0.5) / 0.28209479 * 0.15 + 0.5) * 255), 0, 255).astype(np.uint8)   # SH0（sight.py 讀回來是 rgb）
    n = len(xyz)
    write(out, {'xyz': xyz, 'alpha': np.full(n, 255, np.uint8), 'color': color, 'logs': np.full((n, 3), np.log(SIZE)),
                'quat': np.tile([0, 0, 0, 1.0], (n, 1)), 'frac': FRAC, 'flags': 0})
    print(f'{len(meshes)} 塊網格、表面 {sum(m.area for m in meshes):.0f} m² → {n} 點 → {out}')


if __name__ == '__main__':
    main()
