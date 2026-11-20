#!/bin/zsh
# 路線經過建模真照片對位（用沒裁切的 merged.spz：岩角、突出的岩塊在裁切框外）（sfm_to_model.py 把真照片相機對到模型、photo_real.py 找最像的真照片、photo_pose.py batch 求拍攝位置與投線）
# 用法：photo_real.sh [路線編號…]（預設：清單上全部路線）；結果寫進 photo/，舊結果先備份到 photo-prev/
cd ${0:a:h}; T=../../tools/crag3d; PY=${PY:-python3}
NUMS=${*:-"$(cut -d' ' -f2 photo/todo.txt | tr '\n' ' ')"}
[ -f photo/cameras.json ] || ../dapaoyan-3d/venv/bin/python -I $T/sfm_to_model.py ../dapaoyan-3d/work/sfm/0 ../dapaoyan-3d crop.spz photo/cameras.json || exit 1
mkdir -p photo-prev; for n in ${=NUMS}; do cp photo/$n.json photo/$n.jpg photo-prev/ 2>/dev/null; done
$PY -I $T/photo_real.py crag-sight.json photo/todo.txt photo/walls.txt photo/cameras.json ../dapaoyan-3d/work/img photo ${=NUMS} 2>&1 | grep --line-buffered -v -i warn || exit 1
$PY -I $T/photo_pose.py batch merged.spz crag-sight.json photo/todo-real.txt refs photo 2>&1 | grep --line-buffered -v -i warn
echo 全部完成
