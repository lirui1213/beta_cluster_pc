// Read a trusted saved request; render its summary without starting numerical jobs.
import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {createServer} from "node:http";
import {readFileSync,existsSync,mkdirSync,writeFileSync} from "node:fs";
import {fileURLToPath,pathToFileURL} from "node:url";
import path from "node:path";

const request = process.env.GRID_REQUEST;
assert.ok(request, "Set GRID_REQUEST to a trusted local display request.joblib");
const python = process.env.GRID_PYTHON || "E:/anaconda/envs/ECoG_LR_v1/python.exe";
const code = `
import sys,json,joblib,numpy as np,pandas as pd
sys.path.insert(0,'3-bands')
from pathlib import Path
from fourband.browser_data import session_metadata,cluster_grid_payload
request=joblib.load(sys.argv[1])
meta=session_metadata(request)
grid=cluster_grid_payload(request,meta)
assert grid['available'],grid
frame=pd.read_csv(Path(request['output_dir'])/'fourband_curve_summary.csv')
for panel in grid['panels']:
    rows=frame.loc[(frame.band==panel['band'])&(frame.label==panel['label'])]
    for field,column in [('mean','mean_power_z'),('sem','electrode_sem'),('audio','mean_audio'),('counts','n_electrodes')]:
        np.testing.assert_allclose(np.array(panel[field],float),rows[column].to_numpy(),equal_nan=True)
    np.testing.assert_allclose(panel['times_ms'],rows.display_time_seconds.to_numpy()*1000)
print(json.dumps(dict(grid=grid,subject=meta['subject'],block=meta['block'],rows=len(frame)),allow_nan=False))
`;
const saved = JSON.parse(execFileSync(python,["-c",code,request],{encoding:"utf8",maxBuffer:20*1024**2}));
const assets = path.resolve(fileURLToPath(new URL("../3-bands/fourband/browser_assets/",import.meta.url)));
const output = process.env.FRONTEND_ARTIFACTS || "3-bands/outputs/verification/linked_grid_20261020/actual";
mkdirSync(output,{recursive:true});
const fixture = `import {ClusterGrid} from './cluster_grid.mjs'; import {SessionAPI} from './core.mjs';
const colors=Object.fromEntries(${JSON.stringify(saved.grid.classes)}.map((group,i)=>[group.label,['#167d91','#c45277','#719342','#8a65ab'][i%4]]));
const grid=new ClusterGrid(document.querySelector('#cluster-grid-pane'),()=>colors);
window.gridLoaded=grid.load(new SessionAPI(new URL('./',location.href)));
window.recolorGrid=async(label,color)=>{colors[label]=color;await grid.recolor();};`;
const html = `<!doctype html><html lang="zh"><meta charset="utf-8"><link rel="stylesheet" href="styles.css">
<script defer src="vendor/plotly.js-dist-min/plotly.min.js"></script><script type="module" src="fixture.mjs"></script>
<title>Saved four-band summary verification</title><section id="cluster-grid-pane" class="cluster-grid-pane">
<div class="pane-heading"><h2>四频带分类总览</h2></div><div id="cluster-grid-status">Loading</div><div id="cluster-grid"></div></section></html>`;
const server = createServer((req,res)=>{
  const prefix="/s/saved-fixture/", pathname=new URL(req.url,"http://localhost").pathname;
  if (!pathname.startsWith(prefix)) {res.writeHead(404).end();return;}
  const route = pathname.slice(prefix.length);
  if (!route) {res.setHeader("Content-Type","text/html; charset=utf-8");res.end(html);return;}
  if (route === "api/cluster-grid") {res.setHeader("Content-Type","application/json");res.end(JSON.stringify(saved.grid));return;}
  if (route === "fixture.mjs") {res.setHeader("Content-Type","text/javascript");res.end(fixture);return;}
  const target = path.resolve(assets,route);
  if (!target.startsWith(assets+path.sep) || !existsSync(target)) {res.writeHead(404).end();return;}
  res.setHeader("Content-Type",({".mjs":"text/javascript",".js":"text/javascript",".css":"text/css"})[path.extname(target)] || "application/octet-stream");
  res.end(readFileSync(target));
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
const {chromium} = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE || path.join(process.env.USERPROFILE,".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs")));
let browser;
try {
  browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || "C:/Program Files/Google/Chrome/Application/chrome.exe"});
  const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];
  page.on("pageerror",error=>errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/s/saved-fixture/`);
  await page.waitForFunction(()=>Boolean(window.gridLoaded));
  await page.evaluate(()=>window.gridLoaded);
  assert.equal(await page.locator(".cluster-grid-plot").count(),saved.grid.panels.length);
  const means=await page.locator(".cluster-grid-plot").evaluateAll(nodes=>nodes.map(el=>el.data[2].y));
  const ordered=saved.grid.classes.flatMap(group=>saved.grid.bands.map(band=>saved.grid.panels.find(panel=>panel.label===group.label && panel.band===band.name).mean));
  assert.deepEqual(means,ordered,"Rendered curves exactly match saved CSV");
  await page.screenshot({path:path.join(output,"saved-grid-desktop.png"),fullPage:true});
  const label=saved.grid.classes[0].label;
  await page.evaluate(label=>window.recolorGrid(label,"#d42f66"),label);
  assert.ok(await page.locator(`[data-label="${label}"] .cluster-grid-plot`).evaluateAll(nodes=>nodes.every(el=>el.data[2].line.color==="#d42f66" && el.data[1].fillcolor==="rgba(212,47,102,0.18)")));
  await page.setViewportSize({width:390,height:844});
  await page.waitForFunction(()=>[...document.querySelectorAll(".cluster-grid-plot")].every(el=>Math.abs(el._fullLayout.width-el.clientWidth)<2));
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:path.join(output,"saved-grid-mobile.png"),fullPage:true});
  assert.deepEqual(errors,[]);
  const report={subject:saved.subject,block:saved.block,rows:saved.rows,panels:saved.grid.panels.length,
    csvValuesMatch:true,renderedValuesMatch:true,recolor:true,mobileOverflow:false,errors};
  writeFileSync(path.join(output,"report.json"),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report));
} finally {
  if (browser) await browser.close();
  server.closeAllConnections();
  await new Promise(resolve=>server.close(resolve));
}
