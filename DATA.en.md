# Data

繁體中文：[DATA.md](DATA.md)

Large files live in the Hugging Face dataset [climbio/crag3d-lab](https://huggingface.co/datasets/climbio/crag3d-lab). `data/manifest.json` lists every file (path, size, SHA-256, license, credit) and `tools/download.py` downloads by that list and checks the hashes. Small files (route points, settings, stage outputs, research screenshots) are in this repo.

## Layout of the Hugging Face dataset

| Folder | Contents | Size |
|---|---|---|
| `photos/originals/` | 790 original photos (iPhone originals with EXIF and capture location) | 3.4 GB |
| `photos/originals-20261121/` | 191 photos taken on 2026-11-21 (iPhone originals with EXIF and capture location; for extending the model) | 0.8 GB |
| `guidebook-2d/` | The 2D guidebook as Climbio format v2: `index.json` is the entry, with 336 photo views, the set and the crag map | 451 MB |
| `legacy-2d/` | 85 route photos from the old app (red lines drawn on photos; people blurred in 7), `routes.json`, the area map | 20 MB |
| `models/source/` | Gaussian-splat models made with World Labs Atlas: the whole crag (several versions), one wall, a chimney patch | 233 MB |
| `models/ours/` | Our merged and cropped models (`merged` = whole crag plus that wall, `crop` = cropped to the routes, `crop-show` = the demo version) | 131 MB |
| `models/hybrid/` | Mesh plus splat backdrop (later replaced by the splat version; kept for comparison) | 55 MB |
| `experiments/worldlabs/` | Our three World Labs retrain experiments (`atlas64`, `atlas64-v1`: our own capture positions and depth; `self-pose`: let it estimate positions and depth): images sent, depth (EXR), requests and responses, results, check images (all worse than the Atlas version; kept as evidence of what did not work) | 277 MB |
| `experiments/atlas-full/` | The process for the whole-crag model with World Labs Atlas: the pipeline code (`code/`, step0–4), chosen photos, each round's requests and responses, Atlas-estimated capture positions, depth previews, a sheet of the AI-generated wide views, the raw results of the two training groups | 67 MB |

In the process responses, signed download URLs are cut to their path, and World Labs account IDs, emails and user names in local paths are replaced. Files identical across the three experiments (images sent, depth, check images) are stored once; the manifest points to them with `same_as`, and `tools/download.py` copies them on download.

Not included: the old mesh pipeline output (about 20 GB, rebuildable from the photos), intermediate results and logs, other people's route data, and the old app's database export (it contains other climbers' names).

## `data/dapaoyan/` in this repo

| File | Contents |
|---|---|
| `atlas_routes.json`, `atlas_alignment.json` | Route points in the Atlas model's coordinates and the model alignment |
| `edits.json`, `dapaoyan-edits.json`, `dapaoyan-top-photos.json` | Corrections to route positions (shift, straighten, start and end heights) and the grouping of photos taken from above the wall |
| `crag-*.json`, `drawn*.json` | Routes and walls at each pipeline stage (`crag-base` is the start, `crag-sight` the last one with surface-snapped coordinates) |
| `sight.json` | Occlusion tables |
| `wl-views.json`, `wl2-views.json` | The photo lists sent to World Labs |

## Privacy and location

- Original photos carry capture location (EXIF GPS); the crag location is public information anyway.
- The new photos contain no people; 7 of the old 2D route photos did, and those areas are blurred (`2096031`, `2096032`, `2136022`, `2146017`, `2166026`, `2236010`, `2236011`).
- The old app's database export contains other climbers' names and is not published.

## License

Photos, route drawings, models and other data: CC BY-SA 4.0, credit Climbio (see [CREDITS.md](CREDITS.md)).
