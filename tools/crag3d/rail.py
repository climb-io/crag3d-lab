"""戶外 3D Topo：各區鏡頭沿著岩壁要照順序排，紅點貓左右滑換區時畫面才不會往旁邊跳又跳回來。

鏡頭照岩面朝向（facing.py）擺在正前方；側面（例：砲管1 的陽光面、砲管3、飛鳥）朝向偏 50–64°，鏡頭被擺到很遠的側邊，
沿岩壁的位置前後不連續（大砲岩：新手牆 41 m → 砲管1 48 m → 砲管2 40 m）。這裡只轉鏡頭、不改朝向（貼面、延伸還要用真的法線）：
1. 整段岩壁的正面方向 = 各區朝向的平均。各區鏡頭最多斜 MAX_YAW 度，超過的繞看的中心轉回來（walls[].view.yaw，snap.html 照它擺）。
2. 還是不照順序（沿著第一區到最後一區的方向，位置要一直往前）的，再每次少斜 STEP 度，直到照順序或轉回正面。
鏡頭離看的中心的距離用上一次快照（snaps.json）的；沒有就用 DIST 公尺。

用法：rail.py <crag.json> <snaps.json 或 -> <輸出 crag.json>
"""
import json, math, sys
from pathlib import Path
import numpy as np

MAX_YAW, STEP, DIST = 35, 5, 10.0


def ang(v):
    return math.degrees(math.atan2(-v[0], -v[2]))


def rot(v, deg):
    a = math.radians(deg)
    return np.array([v[0] * math.cos(a) + v[2] * math.sin(a), 0, -v[0] * math.sin(a) + v[2] * math.cos(a)])


def main():
    crag_file, snaps_file, out = sys.argv[1:]
    crag = json.loads(Path(crag_file).read_text())
    snaps = {s['wall']: s for s in json.loads(Path(snaps_file).read_text())} if snaps_file != '-' else {}
    walls = crag['walls']
    F = [np.array(w['facing'], float) * [1, 0, 1] for w in walls]
    g = np.mean([f / np.linalg.norm(f) for f in F], 0); g /= np.linalg.norm(g)
    axis = np.array(walls[-1]['center']) - np.array(walls[0]['center']); axis[1] = 0; axis /= np.linalg.norm(axis)
    dist = []
    for w in walls:
        s = snaps.get(w['name'])
        dist.append(np.linalg.norm(np.array(s['camera']['position']) - np.array(s['camera']['target'])) if s else DIST)
    theta = [((ang(f) - ang(g) + 180) % 360) - 180 for f in F]
    want = [max(-MAX_YAW, min(MAX_YAW, t)) for t in theta]
    def along(i):
        f = F[i] / np.linalg.norm(F[i])
        return (np.array(walls[i]['center']) + rot(f, want[i] - theta[i]) * dist[i]) @ axis
    for _ in range(100):
        a = [along(i) for i in range(len(walls))]
        bad = [i for i in range(len(walls)) if (i and a[i] < a[i - 1]) or (i + 1 < len(walls) and a[i] > a[i + 1])]
        bad = [i for i in bad if want[i] != 0]
        if not bad: break
        i = max(bad, key=lambda k: abs(want[k]))
        want[i] = math.copysign(max(0, abs(want[i]) - STEP), want[i])
    for i, w in enumerate(walls):
        yaw = round(want[i] - theta[i], 1)
        v = dict(w.get('view') or {'yaw': 0, 'scale': 1}); v['yaw'] = round(v['yaw'] + yaw, 1)
        if abs(v['yaw']) > 0.05 or v.get('scale', 1) != 1: w['view'] = v
        print(f"{w['name']}：朝向偏 {theta[i]:+.0f}°，鏡頭斜 {want[i]:+.0f}°，沿岩壁 {along(i):.1f} m")
    Path(out).write_text(json.dumps(crag, ensure_ascii=False))


if __name__ == '__main__':
    main()
