# crag3d-lab

[English](README.en.md) | 繁體中文：[README.md](README.md)

Research notes, code and data for turning a **2D climbing guidebook into a 3D crag topo**: photograph a crag, reconstruct a 3D model, place the routes drawn on the old 2D photos onto the 3D wall, and play it on a phone ([Climbio Redpoint](https://climbio.app)).

**Status: experimental, validated on one crag (Dapaoyan, 大砲岩).** Two of the five stages are solid, one is half-automated, two are still being built (table below). Read this as a **recipe with a full worked example**, not a one-click tool.

## What is here

| Stage | State | Where |
|---|---|---|
| Capture (close-up set + step-back set) | Guide written; the step-back set is only tried on this crag | [SHOOTING.en.md](SHOOTING.en.md) |
| Reconstruction (photos → Gaussian splat) | The best model was made with World Labs Atlas (photos plus AI-generated wide views); two retrains from real photos only were worse (notes in the research log) | `tools/crag3d/atlas_*.py`, [research log](docs/research-log.md) |
| Crop, shrink, occlusion tables, snapshots, packing | Solid; the output plays in Climbio Redpoint | `tools/crag3d/spz_crop.py`, `sight.py`, `snap.html`, `redpoint_pack.py`, … |
| Routes from 2D photos to the 3D wall | Half automatic: project the old photo's red line from its own viewpoint, then check route by route by hand | `tools/crag3d/photo_*.py`, `route_*.py` |
| Publishing and playback | Solid: Climbio format v2 `surface: model`, player `crag3dkit` in the Climbio repo | `data/dapaoyan/`, [format spec](https://github.com/climb-io/climbio/blob/main/format/VOCABULARY.md) |

The scripts that string the stages together for this crag are in `runs/dapaoyan-spz/` (`pack-show.sh` is the one behind the demo; `pack.sh` is the older mesh version). The small inputs and stage outputs (route points, alignment, your edits, occlusion tables) are in `data/dapaoyan/`.

## Data

Photos, models and other large files live in the Hugging Face dataset **[climbio/crag3d-lab](https://huggingface.co/datasets/climbio/crag3d-lab)** (about 4.6 GB; layout in [DATA.en.md](DATA.en.md)). `data/manifest.json` lists every file with size and SHA-256:

```bash
python3 tools/download.py --only models/ours legacy-2d     # pick folders
python3 tools/download.py                                  # everything (about 4.6 GB)
```

## Try the result

The published crag: <https://climbio.app/p/rw9oa7t1taemt9zj> (opens in Redpoint; on a computer use the arrow keys or drag along the wall).

## Run it

Python 3 with `pip install -r requirements.txt`; `snap.html` needs a headless browser (Playwright) and `snap_server.py`. Follow `runs/dapaoyan-spz/pack-show.sh` stage by stage; each tool's header comment says what it takes and gives. Paths in the scripts are the ones we used; `PY` and `LEGACY` can be set in the environment. We have not run the whole thing outside the original machine, so expect to fix paths.

## What we learned

The [research log](docs/research-log.md) (Traditional Chinese, dated) records what we tried, what failed and the numbers. The short version: splats are only sharp near where photos were taken, so a crag needs a step-back photo set; occlusion tables computed on the author side let the phone just play; matching old 2D photos to the 3D model works only from the old photo's own viewpoint.

## License and credits

- Code: AGPL-3.0-or-later ([LICENSE](LICENSE)).
- Photos, route drawings, models and data: CC BY-SA 4.0 ([LICENSE-DATA.md](LICENSE-DATA.md)); credit **Climbio** ([CREDITS.md](CREDITS.md)).
- Seven of the 2D route photos had people in them; those areas are blurred.
