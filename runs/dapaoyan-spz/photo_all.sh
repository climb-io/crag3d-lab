#!/bin/zsh
# 大砲岩 50 條：每區前方一圈參考圖（沒拍齊的補拍）→ 照路線順序對位、投線 → photo/<編號>.json、photo/<編號>.jpg
cd ${0:a:h}; T=../../tools/crag3d; PY=${PY:-python3}; L=${LEGACY:-../../data/hf/legacy-2d}
mkdir -p photo refs snaps
python3 -c "import json; print('\n'.join(w['name'] for w in json.load(open('crag-sight.json'))['walls']))" > photo/walls.txt
i=0
while read w; do
  [ -f photo/views-$i.json ] || $PY -I $T/photo_pose.py views crop.spz crag-sight.json "$w" photo/views-$i.json w$i || exit 1
  if [ $(ls refs/w$i-*.jpg 2>/dev/null | wc -l) -lt 27 ]; then
    cp $T/snap.mjs ../../tools/holds-3d/.snap-run.mjs && node ../../tools/holds-3d/.snap-run.mjs "http://127.0.0.1:8769/snap.html?pack=crag-sight.json&model=crop.spz&views=photo/views-$i.json" | grep -v 404; rm -f ../../tools/holds-3d/.snap-run.mjs
    mv snaps/w$i-*.jpg refs/ || exit 1
  fi
  i=$((i+1))
done < photo/walls.txt
python3 -c "
import json
crag = json.load(open('crag-sight.json')); legacy = {r['number']: r['file'] for r in json.load(open('$L/routes.json'))}
for i, w in enumerate(crag['walls']):
    for r in sorted(crag['routes'], key=lambda r: int(r['number'])):
        if w['routes'][0] <= int(r['number']) <= w['routes'][1] and r.get('points') and r['number'] in legacy: print(i, r['number'], '$L/' + legacy[r['number']])
" > photo/todo.txt
$PY -I $T/photo_pose.py batch crop.spz crag-sight.json photo/todo.txt refs photo 2>&1 | grep -v -i warn
echo 全部完成
