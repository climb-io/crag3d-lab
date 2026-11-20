#!/bin/zsh
# 用法：build.sh <工作資料夾>（含 prep.py 的 work/ 與 sfm 分群後的 groups/g*/photos）；照片多的群也一次只跑一群
T=${0:a:h}; cd $1
for g in groups/g*(N/); do
  [[ -f $g/wall.glb ]] && continue
  mkdir -p $g/model
  print "== $g $(ls $g/photos | wc -l) 張 $(date +%T)"
  /usr/bin/time -l $T/capture_lite $g/photos $g/model 30000 1024 > $g/capture.log 2>&1
  grep -E "相機數|總秒數|maximum resident" $g/capture.log
  ls $g/model/*.obj >/dev/null 2>&1 && python3 -I $T/clean.py $g/model work/prep.json $g/wall.glb 2>&1 | tail -1 | cut -c1-300
done
