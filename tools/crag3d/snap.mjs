// 戶外 3D Topo：用無頭瀏覽器（Playwright 的 Chromium）跑 snap.html，等它把快照存完（標題變「完成」）。
// 用法：NODE_PATH=tools/holds-3d/node_modules node tools/crag3d/snap.mjs <snap_server 的網址，含 snap.html 的參數>
import { chromium } from 'playwright';
const browser = await chromium.launch({ channel: 'chromium', args: ['--use-angle=metal', '--ignore-gpu-blocklist'] });
const page = await browser.newPage();
page.on('console', (m) => m.type() === 'error' && console.error(m.text()));
page.on('pageerror', (e) => console.error(e.message));
await page.goto(process.argv[2]);
await page.waitForFunction(() => document.title === '完成', null, { timeout: 600000, polling: 1000 });
await browser.close();
console.log('完成');
