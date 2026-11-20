#!/bin/zsh
# 補拍沒拍完的參考圖（飛鳥 w8、雙十 w9），再把還沒對位的路線跑完
cd ${0:a:h}; T=../../tools/crag3d; PY=${PY:-python3}; L=${LEGACY:-../../data/hf/legacy-2d}
mkdir -p snaps
for i in 8 9; do
  [ $(ls refs/w$i-*.jpg 2>/dev/null | wc -l) -ge 27 ] && continue
  cp $T/snap.mjs ../../tools/holds-3d/.snap-run.mjs && node ../../tools/holds-3d/.snap-run.mjs "http://127.0.0.1:8769/snap.html?pack=crag-sight.json&model=crop.spz&views=photo/views-$i.json" | grep -v 404; rm -f ../../tools/holds-3d/.snap-run.mjs
  mv snaps/w$i-*.jpg refs/
done
ls refs | sed 's/-.*//' | sort | uniq -c | tr '\n' ' '; echo
while read i n f; do
  [ -f photo/$n.json ] && continue
  echo "== $n"
  $PY -I $T/photo_pose.py solve crop.spz photo/views-$i.json refs $L/$f $n crag-sight.json photo/$n.jpg photo/$n.json 2>&1 | grep -v -i warn | tail -3
done < photo/todo.txt
echo 全部完成
