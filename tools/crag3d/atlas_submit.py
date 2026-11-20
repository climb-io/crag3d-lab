"""戶外 3D Topo：atlas_frames.py 的輸出上傳到 World Labs，組成 posedImages2Splats 請求（實驗）。

要先在終端機裝好、登入 World Labs 指令工具（worldlabs auth login）。照片與深度圖逐張上傳，資產 id 存在 <資料夾>/assets.json，
中斷重跑不會重傳；請求寫到 <資料夾>/request.json。加 --run 才提交（一次 5000 點），提交後等完成並下載到 <資料夾>/out/。

用法：atlas_submit.py <資料夾> [--run]
"""
import json, subprocess, sys
from pathlib import Path

PROMPT = ('Natural weathered grey-brown sandstone rock climbing cliff (Dapaoyan crag in Beitou, northern Taiwan) nestled along a lush green '
          'subtropical forested mountain ridge under soft daylight sky, dense green mountain shrubs, ferns, and trees growing naturally '
          'along the upper cliff crest, side ridges, and rocky sandstone boulder trail at the base.')


def wl(*args):
    r = subprocess.run(['worldlabs', '--json', *args], capture_output=True, text=True)
    if r.returncode: raise SystemExit(f'worldlabs {args[0]} 失敗：{r.stderr or r.stdout}')
    return json.loads(r.stdout)


def main():
    d = Path(sys.argv[1]); run = '--run' in sys.argv
    frames = json.loads((d / 'frames.json').read_text())
    af = d / 'assets.json'; assets = json.loads(af.read_text()) if af.exists() else {}
    for f in frames:
        for key, path, mime in (('img', d / 'img' / f"{f['name']}.jpg", 'image/jpeg'), ('depth', d / 'depth' / f"{f['name']}.exr", 'image/x-exr')):
            k = f"{key}/{f['name']}"
            if k in assets: continue
            info = wl('assets', 'upload', str(path), '--mime', mime)
            assets[k] = info.get('id') or info.get('assetId'); af.write_text(json.dumps(assets, indent=1))
            print('上傳', k, flush=True)
    req = {'prompt': PROMPT, 'posedFrames': [{
        'imageAsset': {'assetId': assets[f"img/{f['name']}"]},
        'depth': {'depthAsset': {'assetId': assets[f"depth/{f['name']}"]}},
        'camera': {'extrinsics': {'position': f['position'], 'quaternion': f['quaternion'], 'coordinateSystem': 'rub'},
                   'intrinsics': f['intrinsics']}} for f in frames]}
    (d / 'request.json').write_text(json.dumps(req, indent=1))
    print(f"請求檔 {len(frames)} 張 → {d / 'request.json'}")
    if not run: return
    opf = d / 'operation.json'
    op = json.loads(opf.read_text()) if opf.exists() else None
    if not op:
        op = wl('tasks', 'run', 'posedImages2Splats', '--request', str(d / 'request.json')); opf.write_text(json.dumps(op, indent=1))
    oid = op.get('id') or op.get('operationId') or op.get('name'); print('已提交', oid, flush=True)
    st = wl('operations', 'wait', oid, '--timeout', '3600')
    if st.get('error'): raise SystemExit(f"失敗：{st['error']}")
    subprocess.run(['worldlabs', 'operations', 'download', oid, '--out', str(d / 'out')], check=True)
    print('完成，下載到', d / 'out')


if __name__ == '__main__':
    main()
