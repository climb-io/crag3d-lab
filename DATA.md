# 資料

English: [DATA.en.md](DATA.en.md)

大檔放在 Hugging Face 資料集 [climbio/crag3d-lab](https://huggingface.co/datasets/climbio/crag3d-lab)，`data/manifest.json` 列出每個檔案（路徑、大小、SHA-256、授權、署名），`tools/download.py` 照清單下載並檢查雜湊。小檔（路線點、設定、各階段輸出、研究截圖）直接在這個 repo。

## Hugging Face 資料集的結構

| 資料夾 | 內容 | 大小 |
|---|---|---|
| `photos/originals/` | 原始照片 790 張（iPhone 原檔，含 EXIF 與拍攝位置） | 3.4 GB |
| `guidebook-2d/` | 2D 路線書：Climbio 格式 v2，`index.json` 是入口，336 張照片的畫面、組合、岩場地圖 | 451 MB |
| `legacy-2d/` | 舊 App 的 2D 路線圖 85 張（照片上畫了紅線；7 張拍到人的部分已模糊）、路線資料 `routes.json`、區域圖 | 20 MB |
| `models/source/` | 用 World Labs Atlas 做的高斯潑濺模型：全區（多個版本）、梁大隆、煙囪補丁 | 233 MB |
| `models/ours/` | 我們合併、裁切後的模型（`merged` 全區加梁大隆、`crop` 裁到路線範圍、`crop-show` 展示版） | 131 MB |
| `models/hybrid/` | 網格加潑濺背景的做法（後來改成潑濺版，留作比較） | 55 MB |
| `experiments/worldlabs/` | 我們用 World Labs 重訓的三次實驗（`atlas64`、`atlas64-v1`：自己算的拍攝位置與深度；`self-pose`：讓它自己估位置與深度）：送出的圖、深度（EXR）、請求與回應、結果、檢查圖（都比 Atlas 版差，留作「踩過的坑」的證據） | 277 MB |
| `experiments/atlas-full/` | 用 World Labs Atlas 做全區模型的過程：流程程式（`code/`，step0–4）、選的照片、每一輪的請求與回應、Atlas 估的拍攝位置、深度預覽、AI 生成的廣角圖拼圖、分兩群訓練的原始結果 | 67 MB |

處理過程的回應檔裡，帶簽章的下載網址只留路徑，World Labs 帳號 ID、email、本機路徑裡的使用者名稱都換掉了。三次實驗之間內容相同的檔案（送出的圖、深度、檢查圖）只放一份，清單裡用 `same_as` 指過去，`tools/download.py` 下載時會複製。

沒有放的：舊的網格版流程輸出（約 20 GB，可由照片重建）、中間結果與紀錄檔、別人的路線資料、舊 App 資料庫的匯出（含其他攀岩者的名字）。

## 這個 repo 的 `data/dapaoyan/`

| 檔案 | 內容 |
|---|---|
| `atlas_routes.json`、`atlas_alignment.json` | Atlas 模型座標裡的路線點與模型對位參數 |
| `edits.json`、`dapaoyan-edits.json`、`dapaoyan-top-photos.json` | 對路線位置的修正（左右移、拉直、起終點高度）、岩壁上方照片的分組 |
| `crag-*.json`、`drawn*.json` | 流程各階段的路線與岩壁（`crag-base` 是起點，`crag-sight` 是最後一份，含貼面後的座標） |
| `sight.json` | 遮擋表 |
| `wl-views.json`、`wl2-views.json` | 送給 World Labs 的照片清單 |

## 隱私與位置

- 原始照片帶拍攝位置（EXIF GPS）；岩場位置本來就是公開資訊。
- 新拍的照片沒有拍到人；舊 2D 路線圖有 7 張拍到人，已模糊（`2096031`、`2096032`、`2136022`、`2146017`、`2166026`、`2236010`、`2236011`）。
- 舊 App 的資料庫匯出含其他攀岩者的名字，不公開。

## 授權

照片、路線圖、模型、程式以外的資料：CC BY-SA 4.0，署名 Climbio（見 [CREDITS.md](CREDITS.md)）。
