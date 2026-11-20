"""戶外 3D Topo：讓 World Labs 自己估拍攝位置與深度，再用它自己的結果訓練潑濺（實驗）。

自己算的位置、網格深度比較準，但不一定最適合 World Labs 訓練（2026-11-16 用自己算的 64 張訓練：畫面蒙霧、拉絲）；
這裡照 Atlas 版的做法，先 images2PosedRGBD（位置、深度、內參都是它自己估的，彼此一致），再 posedImages2Splats，
跑完用自己算的拍攝位置（atlas_frames.py 的 frames.json）把潑濺對齊到網格的座標（相似變換：縮放、旋轉、平移，Umeyama）。

子命令（資料夾是 atlas_frames.py 的輸出：img/*.jpg、frames.json）：
- pose：上傳照片、跑 images2PosedRGBD、下載到 <資料夾>/pose/result/（帳號：目前登入的 worldlabs）
- splats [--skip=照片,…] [--run]：pose/ 的照片、深度重新上傳（可以換帳號）、組 posedImages2Splats 請求（相機用它估的），--run 才提交（5000 點）並下載到 <資料夾>/splats/result/
- align <輸出.spz>：它估的拍攝位置對到自己算的位置，潑濺換到網格的座標
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from atlas_submit import wl, PROMPT


def upload(d, cache, key, path, mime):
    c = json.loads(cache.read_text()) if cache.exists() else {}
    if key not in c:
        info = wl('assets', 'upload', str(path), '--mime', mime); c[key] = info.get('id') or info.get('assetId')
        cache.write_text(json.dumps(c, indent=1)); print('上傳', key, flush=True)
    return c[key]


def run_op(task, req, out):
    opf = out / 'operation-id.json'
    op = json.loads(opf.read_text()) if opf.exists() else None
    if not op:
        out.mkdir(parents=True, exist_ok=True); (out / 'request.json').write_text(json.dumps(req, indent=1))
        op = wl('tasks', 'run', task, '--request', str(out / 'request.json')); opf.write_text(json.dumps(op, indent=1))
    oid = op.get('id') or op.get('operationId'); print(task, '已提交', oid, flush=True)
    st = wl('operations', 'wait', oid, '--timeout', '3600')
    if st.get('error'): raise SystemExit(f"失敗：{st['error']}")
    if not (out / 'result' / 'operation.json').exists():   # 下載的資料夾要是空的
        subprocess.run(['worldlabs', 'operations', 'download', oid, '--out', str(out / 'result')], check=True)
    return json.loads((out / 'result' / 'operation.json').read_text())


def main():
    d, cmd, *rest = sys.argv[1:]; d = Path(d)
    frames = json.loads((d / 'frames.json').read_text())
    if cmd == 'pose':
        req = {'frames': [{'imageAsset': {'assetId': upload(d, d / 'pose-assets.json', f['name'], d / 'img' / f"{f['name']}.jpg", 'image/jpeg')}} for f in frames],
               'targetResolution': [720, 1280]}
        op = run_op('images2PosedRGBD', req, d / 'pose')
        r = op.get('response') or {}
        print('估出', len(r.get('frames', [])), '張；位置微調', r.get('poseRefinementAccepted'))
    elif cmd == 'splats':
        r = json.loads((d / 'pose' / 'result' / 'operation.json').read_text())['response']
        cache = d / 'splats-assets.json'; posed = []
        skip = set(a[len('--skip='):].split(',')) if (a := next((x for x in rest if x.startswith('--skip=')), None)) else set()
        for i, fr in enumerate(r['frames']):
            if frames[i]['name'] in skip: continue   # 它估錯位置的（跟其他照片接不起來，放進去會長出殘影）
            img = upload(d, cache, f'img/{i}', d / 'pose' / 'result' / fr['imageAsset']['path'], 'image/jpeg')
            dep = upload(d, cache, f'depth/{i}', d / 'pose' / 'result' / fr['depth']['depthAsset']['path'], 'image/x-exr')
            posed.append({'imageAsset': {'assetId': img}, 'depth': {'depthAsset': {'assetId': dep}}, 'camera': fr['camera']})
        req = {'prompt': PROMPT, 'posedFrames': posed}
        (d / 'splats-request.json').write_text(json.dumps(req, indent=1)); print('請求', len(posed), '張')
        if '--run' in rest: run_op('posedImages2Splats', req, d / 'splats')
    elif cmd == 'align':
        from spz_crop import read, write, transform
        r = json.loads((d / 'pose' / 'result' / 'operation.json').read_text())['response']
        A = np.array([fr['camera']['extrinsics']['position'] for fr in r['frames']], float)   # 它估的（順序跟上傳一樣）
        B = np.array([f['position'] for f in frames], float)                                    # 自己算的
        def fit(m):
            ma, mb = A[m].mean(0), B[m].mean(0); a, b = A[m] - ma, B[m] - mb
            U, S, Vt = np.linalg.svd(b.T @ a / m.sum()); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
            R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (a ** 2).sum() * m.sum(); return s, R, mb - s * R @ ma
        m = np.ones(len(A), bool)
        for _ in range(6):   # 穩健：估錯的照片（例：偏 20–30 m）不拿來對齊
            s, R, t = fit(m); err = np.linalg.norm((s * A @ R.T + t) - B, axis=1); m = err < max(1.0, 2 * np.percentile(err[m], 75))
        print(f'縮放 {s:.3f}；用 {m.sum()} 張對齊，誤差 中位 {np.median(err[m]):.2f} m、最大 {err[m].max():.2f} m')
        spz = next((d / 'splats' / 'result').glob('*.spz')); sp = read(str(spz))
        transform(sp, {'scale_atlas_to_holdlab': s, 'R_atlas_to_holdlab': R, 't_atlas_to_holdlab': t})
        write(rest[0], sp); (d / 'align.json').write_text(json.dumps({'scale': s, 'R': R.tolist(), 't': t.tolist(), 'err_median': float(np.median(err))}))
        print('→', rest[0])


if __name__ == '__main__':
    main()
