#!/bin/zsh
# 第二輪：對位不穩的區補近距離、更斜、往上看的參考圖，不穩的路線用「原本＋補拍」的參考圖重對一次（第一輪結果留在 photo1/）
cd ${0:a:h}; T=../../tools/crag3d; PY=${PY:-python3}
WEAK="22 23 24 25 26 27 37 38 39 40 44 45 46 47 48 50 51 52 53 54 55 56 57 58 59 60"
mkdir -p photo1 snaps; cp -n photo/[0-9]*.json photo/[0-9]*.jpg photo1/ 2>/dev/null
for i in $(for n in ${=WEAK}; do grep " $n " photo/todo.txt | cut -d' ' -f1; done | sort -u); do
  w=$(sed -n "$((i+1))p" photo/walls.txt)
  [ -f photo/views-n$i.json ] || $PY -I $T/photo_pose.py views crop.spz crag-sight.json "$w" photo/views-n$i.json n$i near || exit 1
  if [ $(ls refs/n$i-*.jpg 2>/dev/null | wc -l) -lt 66 ]; then
    cp $T/snap.mjs ../../tools/holds-3d/.snap-run.mjs && node ../../tools/holds-3d/.snap-run.mjs "http://127.0.0.1:8769/snap.html?pack=crag-sight.json&model=crop.spz&views=photo/views-n$i.json" | grep -v 404; rm -f ../../tools/holds-3d/.snap-run.mjs
    mv snaps/n$i-*.jpg refs/ || exit 1
  fi
  python3 -c "import json; a=json.load(open('photo/views-$i.json'))+json.load(open('photo/views-n$i.json')); json.dump(a,open('photo/views-c$i.json','w'),ensure_ascii=False)"
done
for n in ${=WEAK}; do grep " $n " photo/todo.txt | sed 's/^\([0-9]*\) /c\1 /'; done > photo/todo-near.txt
$PY -I $T/photo_pose.py batch crop.spz crag-sight.json photo/todo-near.txt refs photo 2>&1 | grep -v -i warn
echo 全部完成
