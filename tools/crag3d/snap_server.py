"""戶外 3D Topo：快照頁（snap.html）用的本機伺服器。照常提供工作資料夾的靜態檔，另外收 POST /save/<檔名> 存進輸出資料夾。

用法：snap_server.py <工作資料夾> <輸出資料夾> [埠，預設 8769]；開 http://127.0.0.1:<埠>/snap.html?pack=<crag.json>&model=<glb>
"""
import functools, http.server, sys
from pathlib import Path

root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
port = int(sys.argv[3]) if len(sys.argv) > 3 else 8769
out.mkdir(parents=True, exist_ok=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        name = Path(self.path.removeprefix('/save/')).name   # 只收檔名，不收路徑
        if not self.path.startswith('/save/') or not name:
            return self.send_error(404)
        (out / name).write_bytes(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(204); self.end_headers()


http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(Handler, directory=str(root))).serve_forever()
