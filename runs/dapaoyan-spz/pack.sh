#!/bin/zsh
# 大砲岩 3D（Atlas 版的高斯潑濺全區模型）：裁到梁大隆～雙十 → 各區岩面朝向 → 快照拍各區鏡頭（要先開 crag-snap 伺服器）
# → 照舊照片重描的路線（route_draw.py）→ 鏡頭前面被擋的區轉角度（clearance.py）→ 路線貼面、延伸到岩頂、遮擋表 → 用延伸後的路線重拍快照（2D 線）→ 紅點貓資料包
cd ${0:a:h}; T=../../tools/crag3d; PY=${PY:-python3}
# 快照是手機的比例（1080×2340）：左右的範圍跟原本 3:4、垂直 50° 一樣（水平約 38.6°），上下多留，手機上紅點貓的 3D 打開就是整張快照，小卡直接放大到最後的畫面
# 岩壁用網格（Climbio 用全部 790 張照片建的 v11，hybrid/rock.glb），四周山林天空用挖掉岩壁的潑濺背景（hybrid/bg.spz）；
# 潑濺只用 24 張照片加 AI 生成的圖訓練，砲管等凸出的岩塊失真。計算（朝向、清障、遮擋表、貼面、延伸）用網格表面取的點（mesh_points.py → hybrid/rock-points.spz）
ROCK=hybrid/rock.glb; BG=hybrid/bg2.spz; PTS=hybrid/rock-points.spz
# 背景的來源：全區潑濺（Climbio 用 24 張照片加 AI 生成的廣角圖訓練）。試過重新挑 64 張真照片、自己算位置與深度用 posedImages2Splats 重訓
#（atlas64/out/，atlas_frames.py、atlas_submit.py）：照片幾乎都對著岩壁，天空、岩頂一片拉絲，平台前景破洞，比原本差，不用
BGSRC=crop.spz
snap() { cp $T/snap.mjs ../../tools/holds-3d/.snap-run.mjs && node ../../tools/holds-3d/.snap-run.mjs "http://127.0.0.1:8769/snap.html?pack=$1&model=$ROCK&bg=$BG&w=1080&h=2340&fov=74.36&near=999" | grep -v 404; rm ../../tools/holds-3d/.snap-run.mjs; }
# 全區模型在梁大隆那一段放錯了岩塊（層層疊疊的那塊，不是梁大隆），換成另外做的梁大隆（climber-lab atlas，對位參數 atlas_alignment.json）：x ≥ 55、y ≤ 10.8（岩面）用梁大隆模型，岩頂與破洞用全區模型
$PY -I $T/spz_merge.py dapaoyan_full_raw_v3.spz atlas-dl/liangdalong_splats.spz atlas_alignment.json 55.0 10.8 merged.spz || exit 1
# 舊資料沒有座標的路線（2014 年舊 App 的名稱、難度）先加進來，再照舊照片描線
python3 -c "import json; c=json.load(open('../dapaoyan-3d/final/crag.json')); have={r['number'] for r in c['routes']}; c['routes']+=[dict(number=n, name=a, grade=g, color=k, area=2, points=[]) for n, a, g, k in [('20','下降面岩角','5.6','Green'),('37','砲管左岩角','5.5','Green'),('50','鳥頭','5.6','Green'),('51','飛鳥','5.11b','Red'),('52','飛鳥右','5.11c','Red')] if n not in have]; json.dump(c, open('crag-base.json','w'), ensure_ascii=False)" || exit 1
# 裁切的範圍照路線：先換成Atlas 版的位置（梁大隆舊的位置在錯的岩塊上），再換成舊照片投到模型上的位置（photo_real.sh 的結果），
# 突出的岩塊、岩角（例：50 號沿著飛鳥右邊的岩角）才不會被裁掉
python3 -c "
import json
c = json.load(open('crag-base.json')); fr = {str(r['number']): r['spzPoints'] for r in json.load(open('atlas_routes.json'))['routes']}
for r in c['routes']:
    if r['number'] in fr: r['points'] = fr[r['number']]
json.dump(c, open('crag-crop.json', 'w'), ensure_ascii=False)" || exit 1
$PY -I $T/photo_routes.py crag-crop.json photo crag-crop.json merged.spz || exit 1
$PY -I $T/spz_crop.py merged.spz - crag-crop.json crop.spz || exit 1
# 背景：裁好的潑濺挖掉跟網格重疊、擋在網格前面的點，網格沒有的岩頂、屋簷上方由潑濺補上（Climbio 的背景把岩壁整片挖掉，岩頂會露出天空）
$PY -I $T/carve_bg.py $BGSRC $PTS $BG || exit 1
# 路線照舊照片重描：drawn.json 描在 wide/ 的寬快照上（snap.html 的 pad=4）；drawn2.json 描在 trace/（砲管正面、左側斜 55° 看陽光面、平台、飛鳥、雙十各區鏡頭）
$PY -I $T/route_draw.py crop.spz crag-base.json wide drawn.json crag-drawn0.json || exit 1
$PY -I $T/route_draw.py crop.spz crag-drawn0.json trace drawn2.json crag-drawn1.json route-checks || exit 1
# 路線位置以Atlas 版（climber-lab atlas，跟這個模型同一套座標）的路線點為底，再照常貼面、延伸、整理（使用者指定；對照舊照片與自建的梁大隆模型 E13b）；Atlas 版沒有的（37、50–52）用上面描的
python3 -c "
import json
c = json.load(open('crag-drawn1.json')); fr = {str(r['number']): r['spzPoints'] for r in json.load(open('atlas_routes.json'))['routes']}
for r in c['routes']:
    if r['number'] in fr: r['points'] = fr[r['number']]
json.dump(c, open('crag-drawn1.json', 'w'), ensure_ascii=False)" || exit 1
# 最後換成舊照片投到模型上的位置（舊照片的拍攝角度投線，分得出內角、岩角在哪一面）；拍攝位置不可信、投不到的留上面的
$PY -I $T/photo_routes.py crag-drawn1.json photo crag-drawn1.json merged.spz || exit 1
# 分區：53 號在雙十左邊的內角（舊照片跟 54、55 號同一個角度），歸到雙十；砲管拆成三區，各自一個鏡頭
#（陽光面 34–36 在大岩塊的左側面；37–39 正面；40 右岩角、41–42 在岩角轉過去的那一面）。中心照路線點平均，站位跟原本的砲管同方向
python3 -c "
import json, numpy as np
c = json.load(open('crag-drawn1.json')); w = {x['name']: x for x in c['walls']}
w['飛鳥']['routes'][1] = 52; w['雙十']['routes'][0] = 53
pg = w['砲管']; i = c['walls'].index(pg); subs = []
for name, a, b in [('砲管1', 34, 36), ('砲管2', 37, 39), ('砲管3', 40, 43)]:
    pts = np.array([p for r in c['routes'] if a <= int(r['number']) <= b for p in r.get('points') or []])
    ctr = pts.mean(0); stand = ctr + np.array(pg['standAt']) - np.array(pg['center'])
    subs.append(dict(pg, name=name, routes=[a, b], center=ctr.round(3).tolist(), top=[ctr[0], pg['top'][1], ctr[2]], min=pts.min(0).round(3).tolist(), max=pts.max(0).round(3).tolist(), standAt=stand.round(3).tolist()))
c['walls'][i:i + 1] = subs
json.dump(c, open('crag-walls.json', 'w'), ensure_ascii=False)" || exit 1
$PY -I $T/facing.py $PTS crag-walls.json crag-drawn.json || exit 1
# 使用者的指正（標準答案的一部分，edits.json）：沿岩壁左右移，之後 sight.py 貼回岩面、照名稱定位、整理
python3 -I $T/route_edits.py crag-drawn.json ../../data/crag3d/dapaoyan-edits.json crag-drawn.json || exit 1
snap crag-drawn.json
# 鏡頭前面離鏡頭很近的東西擋著的區（例：飛鳥正前方的岩柱）轉一點角度，再拍一次
$PY -I $T/clearance.py $PTS crag-drawn.json snaps/snaps.json crag-clear.json || exit 1
# 各區鏡頭沿岩壁照順序排（側面的區鏡頭最多斜 35°），紅點貓左右滑換區時畫面不會往旁邊跳又跳回來
python3 -I $T/rail.py crag-clear.json snaps/snaps.json crag-clear.json || exit 1
snap crag-clear.json
python3 ../../tools/memguard.py --limit 3000 -- $PY -I $T/sight.py $PTS crag-clear.json snaps/snaps.json sight.json crag-sight.json | grep -v 仰角 || exit 1
snap crag-sight.json
$PY -I $T/redpoint_pack.py crag-sight.json snaps $ROCK pack 121.5199,25.1424 sight.json --book=rw9oa7t1taemt9zj --background=$BG || exit 1
# 放大看會不會斷線、有沒有怪線（有問題只列出來，逐條放大截圖核對）
python3 -I $T/route_check.py pack
