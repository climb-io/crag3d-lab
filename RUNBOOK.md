# 天然岩場 3D 路線書：模型更新後重做與發佈的規範（大砲岩）

給任何代理（Claude Code、Codex、其他引擎）照著做：新的或擴建的 3D 模型進來之後，自動做出 3D 路線書，驗收過再發佈。過去的經驗與數字在 [docs/research-log.md](docs/research-log.md)（「高斯潑濺模型取代網格」「舊路線照片自動投到 3D」「展示版」「路線貼面的修正」幾節）；資料在 Hugging Face `climbio/crag3d-lab`。

## 產品規則（使用者定的，不要改）

- **一個岩場一個模型**，作者端算好全部（鏡頭、遮擋表、路線貼面），手機（紅點貓）只播放。讀者端的行為見 Climbio 的 `docs/redpoint.md` 與 `crag3dkit/`。
- **模型只能由照片合成**：不准推算、合成照片裡沒有的表面和顏色。大洞先找照片或補拍；照片沒拍到就照實留空並回報。小洞可補，但目前暫停。
- **範圍是一段連續的岩壁**，看起來自然；前景草葉、天空碎點在作者端裁掉。
- **路線**：
  - 全部是頂繩，用紅色，要延伸到岩頂，不能停在半途。
  - 位置以 2014 年舊路線照片為準：用舊照片自己的視角投線（`photo_routes.py`），只有低分的才給人看。
  - 使用者的指正 `data/dapaoyan/dapaoyan-edits.json` 是標準答案，每次都要套。
- **各區鏡頭**：
  - 正對岩面法線（`facing.py`），用路線點附近 0.6 m 的法線，不用拍照站位。
  - 沿岩壁照順序排，側面最多斜 35°（`rail.py`）。
  - 鏡頭前面被近物擋住就轉角度（`clearance.py`）。
- **零手動**：不確定的照片與資料自動丟掉，不要求人修。
- **潑濺要退後拍**：不要拿貼近岩壁拍的照片重訓背景，2026-11-16、17 兩次都比原版差。
- **大小**：手機展示模型 ≤ 25 MB，用 `spz_crop.py` 的 `MAX_POINTS` 控制。
- **書 id 沿用 `rw9oa7t1taemt9zj`**（`redpoint_pack.py --book=…`）。id 換了，收過的人不會更新。
- **授權**：CC BY-SA 4.0，署名 Climbio。

## 現況（2026-11-22）

- **線上是第 22 版**：只展示到新手牆，4 區、6–33 號、24 條，流程是 `runs/dapaoyan-spz/pack-show.sh`。
  - 計算用 `crop.spz`，展示用 `crop-show.spz`（24.5 MB）。
  - 模型是全區潑濺 `dapaoyan_full_raw_v3.spz`；梁大隆段換成 `liangdalong_splats.spz`（`spz_merge.py`：x ≥ 55、y ≤ 10.8，對位參數 `atlas_alignment.json`）。
- **十字、砲管（拆三區）、飛鳥、雙十**：路線位置已有（`crag-sight.json`），但當時模型品質不夠，沒上線。
- **2026-11-21 補拍 191 張**（iPhone 12，07:55–08:13），在 Hugging Face 的 `photos/originals-20261121/`。
  - 挑照片用 `tools/crag3d/select_cover.py`：照岩面涵蓋度挑。
  - 砲管試跑：對齊殘差 3.7 cm，挑 170 張，各面沒照片看到的點 ≤ 1.4%。
- **本機中間檔遺失**：2026-11-21 作者機器上的中間檔遺失，中間檔改從 Hugging Face `climbio/crag3d-lab` 下載。
  - `models/source/`：原始模型。
  - `models/ours/`：`merged`、`crop`、`crop-show`。
  - `models/hybrid/`：`rock-points.spz`、`bg-carved.spz`。
  - 各步驟的 JSON 在 GitHub `climb-io/crag3d-lab` 的 `data/dapaoyan/`。
  - 下載用該 repo 的 `tools/download.py`，會驗 sha256。
  - **沒公開、已遺失的**：`route_draw.py` 用的 `wide/`、`trace/` 快照，和 `photo/` 的中間站相機（`cameras-splat.json`）。所以不重描路線，沿用已有的路線點；新區或模型變了的區改用 `photo_routes.py`。

## 新模型進來後的步驟

1. **對座標**：新模型要放到現在路線的座標（`crag-sight.json` 的路線點、`rock-points.spz`）。
   - 先 ICP 對齊。岩壁是直的，ICP 抓不到上下滑：之前整個模型被墊高約 0.5 m，表面距離卻只有 5 cm。
   - 所以要另外用已知相機拍快照、跟真照片比對量上下左右的錯開（`photo_check.py` 的做法），錯開要 < 0.1 m。
   - 換算參數存成 JSON，數字寫進研究紀錄。
2. **換模型**：`pack-show.sh` 的 `merged.spz` 改成新模型。新模型的梁大隆段如果已經正確，拿掉 `spz_merge.py` 那步；否則照舊合併。
3. **路線**：
   - 沿用 `crag-drawn1.json` 的路線點。
   - 新增的區，或模型在那一區明顯變了，跑 `photo_routes.py`（舊照片投線）。寫入條件：PnP 內點 ≥ 300、佔配對 ≥ 18%、紅線打到模型 ≥ 60%、拍攝位置前方 1.5 m 內被擋 ≤ 10%；不過的留原本的點。
   - 一定套 `dapaoyan-edits.json`。
4. **展示範圍**：`pack-show.sh` 裡的區清單和 `spz_crop.py --to 新手牆`，改成這次要上線的最後一區（全部做完就是雙十）。展示模型超過 25 MB 就調 `MAX_POINTS`。
5. **跑完整條流程**：`facing` → 快照 → `clearance` → `rail` → 快照 → `sight`（用 `memguard.py` 限記憶體）→ 快照 → `redpoint_pack` → `route_check`。
   - 快照要先開 `snap_server.py <工作資料夾> <輸出資料夾> 8769`，`snap.mjs` 要 Node 與 playwright。
6. **8 GB 的 Mac 一次只跑一個重工作**（建模、`sight.py`、全批投線），不要並行。

## 驗收（全部過才算完成）

- `route_check.py pack` 沒有斷線、怪線。有問題就逐條放大截圖核對，自己定案，不留給人複核。
- `sight.py` 的報告：各區正面鏡頭看得到的路線點 ≥ 89%，跟第 22 版相同或更好。
- 展示模型 ≤ 25 MB。每一區的快照：
  - 畫面滿版；
  - 沒有大破洞；
  - 前景沒擋住岩面；
  - 路線到岩頂；
  - 共用段座標相同（分叉、合併規則見 Climbio 的 `format/VOCABULARY.md`）。
- 每區一張快照截圖，附在追蹤的 issue。
- 數字、用的指令寫進研究紀錄。
- 公開資料要同步：新模型與中間檔更新 `climbio/crag3d-lab` 與 `climb-io/crag3d-lab`。

## 發佈

`redpoint_pack.py` 的輸出就是 Climbio 開放格式 v2 的資料包（`surface: model`、`props.model.camera`、`props.sight`），用 Climbio Studio 發佈；每一版都要人核准，發佈就是公開、收不回。發佈後讀線上的 `/p/<id>/index.json` 核對名稱、署名、授權，在紅點貓打開確認每區都畫得出來。
