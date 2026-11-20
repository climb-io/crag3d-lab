"""照片上畫好的路線（紅色虛線）→ 由下往上的折線（像素）。"""
import cv2, numpy as np


LINK, SEG = 0.06, 0.02


def trace(bgr):
    """紅色虛線 → 由下往上的折線（像素）；找不到回 None"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h <= 8) | (h >= 172)) & (s > 140) & (v > 90)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    if n < 2: return None
    big = 1 + np.argmax(st[1:, cv2.CC_STAT_HEIGHT])
    # 虛線斜著走、轉彎時，閉運算接不起來，會斷成好幾段：從最高的那段往上下接，端點相距 LINK 比例的高度內、夠長的段都算同一條線
    keep, link = {big}, LINK * bgr.shape[0]
    ends = {i: (np.argwhere(lab == i)[[0, -1]]) for i in range(1, n) if st[i, cv2.CC_STAT_HEIGHT] >= SEG * bgr.shape[0]}
    grew = True
    while grew:
        grew = False
        for i, e in ends.items():
            if i not in keep and any(np.linalg.norm(a - b) < link for k in keep if k in ends for a in e for b in ends[k]): keep.add(i); grew = True
    if sum(st[i, cv2.CC_STAT_HEIGHT] for i in keep) < 0.25 * bgr.shape[0]: return None
    ys, xs = np.nonzero(np.isin(lab, list(keep)))
    pts = [(float(np.median(xs[ys == y])), float(y)) for y in range(ys.max(), ys.min() - 1, -6) if (ys == y).any()]
    return np.array(pts)



def wall_run(X, jump=1.0):
    """打到模型的 3D 點（沒打到是 nan）→ 最長的連續一段與它佔全部點的比例。
    紅線貼著岩角時，有些視線擦過岩角打到後面的岩壁，3D 線會來回跳；在沒打到或跳超過 jump 公尺的地方切開。"""
    best, start = (0, 0), None
    for i in range(len(X) + 1):
        ok = i < len(X) and not np.isnan(X[i, 0])
        if ok and start is not None and np.linalg.norm(X[i] - X[i - 1]) > jump: ok = False; brk = True
        else: brk = False
        if not ok and start is not None:
            if i - start > best[1] - best[0]: best = (start, i)
            start = i if brk else None
        elif ok and start is None: start = i
    return X[best[0]:best[1]], (best[1] - best[0]) / len(X)
