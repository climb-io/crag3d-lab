"""戶外 3D Topo：岩壁用網格時，從高斯潑濺挖出背景（紅點貓的 props.background）。

只挖掉跟網格重疊或擋在網格前面的潑濺點，其餘留下：網格沒有的地方（岩頂、屋簷上方，地面往上拍不到）由潑濺補上，不會露出天空。
每個潑濺點找最近的網格表面點（mesh_points.py 的點和法線）：
- 離網格 SAME 公尺內：同一片岩面，挖掉（網格比較準）。
- 在網格前面（沿法線 0～FRONT 公尺）而且就在那片岩面正前方（離法線那條線 SIDE 公尺內）：潑濺飄在岩壁前面的假凸塊、樹葉，會擋住網格，挖掉。
- 其他（網格後面、旁邊沒有網格的岩頂、四周山林）：留著。

用法：carve_bg.py <潑濺.spz> <網格點.spz（含 .normals.npy）> <輸出.spz>
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read, write

SAME, FRONT, SIDE = 0.3, 3.0, 0.4


def main():
    src, pts_file, out = sys.argv[1:]
    s, m = read(src), read(pts_file)
    M, N = m['xyz'], np.load(pts_file + '.normals.npy').astype(np.float32)
    d, j = cKDTree(M).query(s['xyz'], distance_upper_bound=FRONT + SIDE)
    near = np.isfinite(d); j = np.where(near, j, 0)
    v = s['xyz'] - M[j]; along = (v * N[j]).sum(1); side = np.linalg.norm(v - along[:, None] * N[j], axis=1)
    same = near & (d < SAME)
    front = near & (along > 0) & (along < FRONT) & (side < SIDE)
    drop = same | front
    keep = ~drop
    for k in ('xyz', 'alpha', 'color', 'logs', 'quat'): s[k] = s[k][keep]
    write(out, s)
    print(f'{len(keep)} 點：跟網格重疊 {same.sum()}、擋在前面 {(front & ~same).sum()}，留下 {keep.sum()} → {out}（{Path(out).stat().st_size / 1e6:.1f} MB）')


if __name__ == '__main__':
    main()
