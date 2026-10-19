// Real-data browser acceptance. Run after verify_workspace_analysis.py completes.
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {mkdirSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import {pcaRows, pcaTraces, reviewData} from '../3-bands/fourband/browser_assets/workspace_core.mjs';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {PNG}=require(process.env.PNGJS_MODULE || 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pngjs');
const base=process.env.BROWSER_URL;
assert.ok(base,'Set BROWSER_URL');
const output=process.env.BROWSER_ARTIFACTS || '3-bands/outputs/verification/workspace_actual';
mkdirSync(output,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || 'C:/Program Files/Google/Chrome/Application/chrome.exe',args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1600,height:1000}});
const errors=[],external=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
page.on('request',r=>{if(!r.url().startsWith(new URL(base).origin)&&!r.url().startsWith('data:')&&!r.url().startsWith('blob:'))external.push(r.url());});
const report={};
async function reviewPixels(){
  const image=PNG.sync.read(await page.locator('#pca-plot canvas').first().screenshot());
  let colored=0;
  for(let i=0;i<image.data.length;i+=4){
    const [r,g,b]=image.data.subarray(i,i+3);
    if(Math.max(r,g,b)-Math.min(r,g,b)>90)colored++;
  }
  assert.ok(colored>100,'3D PCA canvas has colored electrode pixels');
  return colored;
}
async function pixelCheck(frame){return frame.evaluate(()=>{
  const source=document.querySelector('#brain canvas'),copy=document.createElement('canvas');
  if(!source)return {colors:0,nonbackground:0};copy.width=source.width;copy.height=source.height;
  const ctx=copy.getContext('2d');ctx.drawImage(source,0,0);const data=ctx.getImageData(0,0,copy.width,copy.height).data;
  let nonbackground=0;const colors=new Set();
  for(let i=0;i<data.length;i+=32){colors.add(`${data[i]},${data[i+1]},${data[i+2]}`);if(Math.abs(data[i]-data[0])+Math.abs(data[i+1]-data[1])+Math.abs(data[i+2]-data[2])>35)nonbackground++;}
  return {colors:colors.size,nonbackground};
});}
async function readyFrame(){
  await page.locator('#tab-results').click();
  await page.waitForFunction(()=>document.querySelector('#explorer')?.getAttribute('src'));
  const frame=page.frameLocator('#explorer');
  await frame.locator('#status-text').filter({hasText:'Ready'}).waitFor({timeout:180000});
  return page.frames().find(f=>f.url().includes('/explorer/'));
}
try{
  await page.goto(base);
  await page.waitForFunction(()=>document.querySelector('#analysis-dialog')?.open || !document.querySelector('#subject')?.disabled);
  if(!await page.locator('#run-id').textContent() && !await page.locator('#analysis-dialog').isVisible()){
    await page.locator('#subject').selectOption('HS0015');
    await page.waitForFunction(()=>document.querySelector('#analysis-dialog')?.open || !document.querySelector('#block')?.disabled);
    if(await page.locator('#block').inputValue()!=='B03') await page.locator('#block').selectOption('B03');
    await page.locator('#analysis-dialog').waitFor({state:'visible'});
  }
  if(await page.locator('#analysis-dialog').isVisible()) await page.locator('#analysis-continue').click();
  await page.waitForFunction(()=>document.querySelector('#run-id')?.textContent.includes('_web_'),{},{timeout:60000});
  report.subject=await page.locator('#subject').inputValue();report.block=await page.locator('#block').inputValue();
  assert.equal(report.subject,'HS0015');assert.equal(report.block,'B03');
  const frame=await readyFrame();
  report.desktopPixels=await pixelCheck(frame);assert.ok(report.desktopPixels.colors>50&&report.desktopPixels.nonbackground>1000);
  report.plots=await frame.evaluate(()=>[...document.querySelectorAll('.js-plotly-plot')].map(p=>({samples:p.data?.[0]?.x?.length,audio:p.data?.some(t=>t.yaxis==='y2')})));
  assert.equal(report.plots.length,4);assert.ok(report.plots.every(p=>p.samples>100&&p.audio));
  await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
  await page.locator('#tab-review').click();
  await page.waitForFunction(()=>document.querySelector('#pca-plot')?.data?.length);
  const accepted=await (await page.request.get(base+'api/workspace/state')).json();
  const expected=pcaTraces(pcaRows(reviewData(accepted))).map(t=>({x:t.x,y:t.y,z:t.z,channels:t.customdata}));
  const actual=await page.locator('#pca-plot').evaluate(el=>el.data.map(t=>({type:t.type,x:t.x,y:t.y,z:t.z,channels:t.customdata})));
  assert.ok(actual.every(t=>t.type==='scatter3d'));
  assert.deepEqual(actual.map(({type,...t})=>t),expected,'All three plotted coordinates match accepted FIRST model');
  report.reviewElectrodes=actual.reduce((n,t)=>n+t.x.length,0);
  report.reviewDesktopPixels=await reviewPixels();
  report.reviewLabels=await page.locator('#bad-clusters option').allTextContents();
  assert.equal(report.reviewLabels.length,3,'Review must use first K=3, not displayed second K=2');
  await page.screenshot({path:path.join(output,'review.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  report.reviewMobilePixels=await reviewPixels();
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
  await page.screenshot({path:path.join(output,'review_mobile.png'),fullPage:true});
  await page.setViewportSize({width:1600,height:1000});
  await page.locator('#tab-trials').click();
  assert.ok(await page.locator('#trial-grid [data-trial]').count()>100);
  await page.screenshot({path:path.join(output,'trials.png'),fullPage:true});
  await page.locator('#tab-settings').click();
  await page.locator('#subject').selectOption('HS0012');
  await page.waitForFunction(()=>document.querySelector('#block')?.options.length>0);
  await page.waitForFunction(()=>!document.querySelector('#explorer')?.getAttribute('src'));
  assert.equal(await page.locator('#explorer').getAttribute('src'),null,'Patient switch must detach old explorer');
  await page.screenshot({path:path.join(output,'other_patient.png'),fullPage:true});
  await page.locator('#subject').selectOption('HS0015');
  await page.locator('#block').selectOption('B03');
  await page.locator('#analysis-dialog').waitFor({state:'visible'});
  await page.screenshot({path:path.join(output,'continue_dialog.png'),fullPage:true});
  await page.locator('#analysis-continue').click();
  await readyFrame();
  await page.setViewportSize({width:390,height:844});
  await page.waitForTimeout(500);
  report.mobileOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
  assert.equal(report.mobileOverflow,false);
  const mobileFrame=page.frames().find(f=>f.url().includes('/explorer/'));
  report.mobilePixels=await pixelCheck(mobileFrame);assert.ok(report.mobilePixels.colors>50&&report.mobilePixels.nonbackground>200);
  await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
  assert.deepEqual(external,[]);assert.deepEqual(errors,[]);
  report.errors=errors;report.external=external;report.verified=true;
  writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report));
}finally{await browser.close();}
