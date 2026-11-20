"""戶外 3D Topo：把另外拍的一段岩壁（高斯潑濺 .spz）換進全區模型。

大砲岩的全區模型在梁大隆那一段是錯的（放的是另一塊層層疊疊的岩塊，跟舊照片不一樣），另外做了梁大隆
（climber-lab atlas 的 liangdalong_splats.spz，用 atlas_alignment.json 對到全區的座標）。
換的範圍：x ≥ X0、而且 y ≤ Y0（梁大隆的岩面；岩頂的草叢、樹用全區模型，另外拍的那份頂部不完整）。
範圍裡全區模型的點，附近 FILL 公尺內沒有另外拍的點（另外拍的那份有破洞，例：梁大隆右邊的岩角）就留著補洞。

用法：spz_merge.py <全區.spz> <另外拍的.spz> <對位 JSON> <X0> <Y0> <輸出.spz>
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
from spz_crop import read, write, transform


FILL = 0.3


def main():
    base_file, patch_file, align_file, x0, y0, out = sys.argv[1:]
    x0, y0 = float(x0), float(y0)
    base, patch = read(base_file), read(patch_file)
    transform(patch, json.loads(Path(align_file).read_text()))
    inside = lambda p: (p[:, 0] >= x0) & (p[:, 1] <= y0)
    b = inside(patch['xyz'])
    a = ~inside(base['xyz'])
    hole = ~a
    hole[hole] = cKDTree(patch['xyz'][b]).query(base['xyz'][hole], distance_upper_bound=FILL)[0] == np.inf
    a |= hole
    merged = dict(base, frac=min(base['frac'], patch['frac']))
    for k in ('xyz', 'alpha', 'color', 'logs', 'quat'):
        merged[k] = np.concatenate([base[k][a], patch[k][b]])
    write(out, merged)
    print(f'全區 {a.sum()} 點（其中補洞 {hole.sum()}）+ 換進來 {b.sum()} 點（x ≥ {x0}、y ≤ {y0}）→ {out}')


if __name__ == '__main__':
    main()
