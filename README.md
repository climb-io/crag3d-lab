# crag3d-lab

繁體中文 | [English](README.en.md)

把**2D 路線書變成 3D 版 Topo** 的研究紀錄、程式與資料：拍一個岩場、重建 3D 模型、把舊 2D 照片上畫的路線放到 3D 岩壁上、在手機上播放（[Climbio 紅點貓](https://climbio.app)）。

**狀態：實驗版，只驗證過一個岩場（大砲岩）。** 五個階段裡兩個穩定、一個半自動、兩個還在做（見下表）。請當成**附完整範例的做法**，不是一鍵可用的工具。

## 這裡有什麼

| 階段 | 狀態 | 位置 |
|---|---|---|
| 拍照（貼近拍加退後拍） | 指南已寫；退後拍只在這個岩場試過 | [SHOOTING.md](SHOOTING.md) |
| 重建（照片 → 高斯潑濺） | 效果最好的是用 World Labs Atlas 做的模型（照片加 AI 生成的廣角圖）；只用真實照片重訓兩次都比較差（寫在研究紀錄） | `tools/crag3d/atlas_*.py`、[研究紀錄](docs/research-log.md) |
| 裁切、壓小、遮擋表、快照、打包 | 穩定；輸出可以在 Climbio 紅點貓播放 | `tools/crag3d/spz_crop.py`、`sight.py`、`snap.html`、`redpoint_pack.py` 等 |
| 路線從 2D 照片放到 3D 岩壁 | 半自動：用舊照片自己的視角把紅線投到模型上，再逐條人工校對 | `tools/crag3d/photo_*.py`、`route_*.py` |
| 發佈與播放 | 穩定：Climbio 格式 v2 的 `surface: model`，播放器 `crag3dkit` 在 Climbio repo | `data/dapaoyan/`、[格式規格](https://github.com/climb-io/climbio/blob/main/format/VOCABULARY.md) |

把各階段串起來的腳本在 `runs/dapaoyan-spz/`（`pack-show.sh` 是展示版用的，`pack.sh` 是較早的網格版）。小的輸入與各階段的輸出（路線點、對位、你的修正、遮擋表）在 `data/dapaoyan/`。

## 資料

照片、模型等大檔放在 Hugging Face 資料集 **[climbio/crag3d-lab](https://huggingface.co/datasets/climbio/crag3d-lab)**（約 4.6 GB，結構見 [DATA.md](DATA.md)）。`data/manifest.json` 列出每個檔案的大小與 SHA-256：

```bash
python3 tools/download.py --only models/ours legacy-2d     # 挑資料夾下載
python3 tools/download.py                                  # 全部（約 4.6 GB）
```

## 看成果

已發佈的岩場：<https://climbio.app/p/rw9oa7t1taemt9zj>（用紅點貓打開；電腦上用方向鍵或沿岩壁拖曳）。

## 自己跑

Python 3，`pip install -r requirements.txt`；`snap.html` 要用無頭瀏覽器（Playwright）加 `snap_server.py`。照 `runs/dapaoyan-spz/pack-show.sh` 一段一段做，每支工具檔案開頭的註解寫了它吃什麼、吐什麼。腳本裡的路徑是我們用的那組，`PY`、`LEGACY` 可以用環境變數換。整套還沒有在原本那台電腦以外的地方完整跑過，預期要自己修路徑。

## 學到什麼

[研究紀錄](docs/research-log.md)（繁體中文，有日期）寫了試過什麼、哪裡失敗、數字。簡短版：潑濺只有在拍照位置附近才清楚，所以岩場要另拍一套退後的照片；遮擋表在作者端算好，手機只要播放；舊 2D 照片對到 3D 模型，只有從舊照片自己的視角投影才對得準。

## 授權與署名

- 程式：AGPL-3.0-or-later（[LICENSE](LICENSE)）。
- 照片、路線圖、模型與資料：CC BY-SA 4.0（[LICENSE-DATA.md](LICENSE-DATA.md)），署名 **Climbio**（[CREDITS.md](CREDITS.md)）。
- 舊 2D 路線圖有 7 張拍到人，那個部分已經模糊。
