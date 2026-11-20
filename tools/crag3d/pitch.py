"""每張照片的俯仰角（iPhone MakerNote 的重力加速度向量），寫回 prep.json 的 pitch（度，正值＝往下拍）。"""
import json, math, struct, sys
from pathlib import Path
from PIL import Image

SRC, W = Path(sys.argv[1]), Path(sys.argv[2])


def accel(p):
    mn = Image.open(p).getexif().get_ifd(0x8769).get(0x927c)
    if not mn or not mn.startswith(b'Apple iOS'): return None
    # 'Apple iOS\0' + 版本 2 bytes + 'MM'，IFD 從 14 開始、大端序，偏移量從 MakerNote 開頭算；tag 8 是加速度向量（3 個 SRATIONAL）
    for i in range(struct.unpack('>H', mn[14:16])[0]):
        tag, typ, cnt, off = struct.unpack('>HHII', mn[16 + 12 * i:28 + 12 * i])
        if tag == 8 and typ == 10 and cnt == 3:
            return [a / b for a, b in (struct.unpack('>ii', mn[off + 8 * k:off + 8 * k + 8]) for k in range(3))]


rows = json.loads((W / 'prep.json').read_text())
for r in rows:
    a = accel(next(SRC.glob(r['name'] + '.*')))
    r['pitch'] = round(math.degrees(math.asin(max(-1, min(1, -a[2] / math.hypot(*a))))), 1) if a else None
(W / 'prep.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
