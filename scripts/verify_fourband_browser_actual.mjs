// Exercise the actual saved-data browser service, not synthetic anatomy.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.BROWSER_URL;
assert.ok(base, 'Set BROWSER_URL to the owned local session URL');
const output = process.env.BROWSER_ARTIFACTS || '3-bands/outputs/verification/browser_actual';
mkdirSync(output, {recursive:true});
const browser = await chromium.launch({headless:true, executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  args:['--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
const page = await browser.newPage({viewport:{width:1600,height:1000}});
const errors=[], external=[];
page.on('pageerror', error=>errors.push(error.message));
page.on('console', message=>{if(message.type()==='error')errors.push(message.text());});
page.on('request', request=>{if(!request.url().startsWith(new URL(base).origin) && !/^(data:|blob:)/.test(request.url()))external.push(request.url());});
async function ready(){await page.waitForFunction(()=>document.querySelector('#status-text')?.textContent==='Ready',{},{timeout:180000});}
try {
  await page.goto(base);
  await ready();
  await page.waitForSelector('#brain canvas');
  await page.waitForFunction(()=>document.querySelector('#mapped-count').textContent.includes('128'));
  await page.locator('#channel').selectOption('ECoG_16');
  await ready();
  await page.waitForFunction(()=>document.querySelector('#data-channel').textContent.includes('ECoG_16'));
  assert.ok((await page.locator('#pair-description').textContent()).includes('5'));
  const pixels=await page.evaluate(()=>{
    const source=document.querySelector('#brain canvas'),copy=document.createElement('canvas');
    copy.width=source.width;copy.height=source.height;
    const context=copy.getContext('2d');context.drawImage(source,0,0);
    const data=context.getImageData(0,0,copy.width,copy.height).data;
    let nonbackground=0;const colors=new Set();
    for(let i=0;i<data.length;i+=16){colors.add(`${data[i]},${data[i+1]},${data[i+2]}`);if(Math.abs(data[i]-data[0])+Math.abs(data[i+1]-data[1])+Math.abs(data[i+2]-data[2])>35)nonbackground++;}
    return {colors:colors.size,nonbackground};
  });
  assert.ok(pixels.colors>50 && pixels.nonbackground>1000,'Real cortex canvas must not be blank');
  const plots=await page.evaluate(()=>[...document.querySelectorAll('.js-plotly-plot')].map(plot=>({
    samples:plot.data[0].x.length, traces:plot.data.length, audio:plot.data.some(trace=>trace.yaxis==='y2'),
    finite:plot.data[0].y.filter(Number.isFinite).length, title:plot.layout.title.text})));
  assert.equal(plots.length,4);assert.ok(plots.every(p=>p.samples>100 && p.finite>100 && p.audio));
  await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
  const canvas=page.locator('#brain canvas'),before=await canvas.screenshot(),box=await canvas.boundingBox();
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();
  await page.mouse.move(box.x+box.width/2+80,box.y+box.height/2+30,{steps:10});await page.mouse.up();
  assert.notDeepEqual(await canvas.screenshot(),before,'Camera movement changes real rendered pixels');
  await page.locator('#reset-camera').click();await page.locator('#labels').check();
  await page.waitForFunction(()=>[...document.querySelectorAll('.electrode-label')].some(label=>!label.hidden));
  const candidates=await page.evaluate(()=>[...document.querySelectorAll('.electrode-label')].filter(label=>!label.hidden && label.textContent!=='ECoG_16').map(label=>{
    const box=label.getBoundingClientRect();return {channel:label.textContent,x:box.x+box.width/2,y:box.y-6};
  }));
  assert.ok(candidates.length,'Visible labelled contacts are required for the pick test');
  let candidate=null;
  // Dense grids can have overlapping sphere glyphs: the foreground contact wins.
  for(const contact of candidates.slice(0,12)){
    await page.mouse.click(contact.x,contact.y);await page.waitForTimeout(100);
    const selected=await page.locator('#channel').inputValue();
    if(selected!=='ECoG_16'){candidate=selected;break;}
  }
  assert.ok(candidate,'Clicking actual visible contacts must select another real channel');
  await ready();
  await page.locator('#channel').selectOption('ECoG_16');await ready();
  await page.locator('#class-filter').selectOption('3');await ready();
  await page.locator('#class-color').fill('#d42f66');await page.locator('#brain-opacity').fill('0.7');
  await page.locator('#labels').uncheck();
  await page.locator('#view-detail').click();
  await page.waitForFunction(()=>document.querySelector('#detail-image').naturalWidth>0 && !document.querySelector('#detail-image').hidden,{},{timeout:180000});
  await ready();
  await page.screenshot({path:path.join(output,'detail.png'),fullPage:true});
  await page.locator('#view-continuous').click();await ready();
  await page.locator('#export').click();
  await page.waitForFunction(()=>document.querySelector('#notice').textContent.includes('Export'),{},{timeout:30000});
  const exportMessage=await page.locator('#notice').textContent();
  await page.reload();await ready();
  assert.equal(await page.locator('#channel').inputValue(),'ECoG_16');
  assert.equal(await page.locator('#class-filter').inputValue(),'3');
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(300);
  const narrow=await page.evaluate(()=>({width:document.documentElement.scrollWidth,viewport:innerWidth,
    brain:document.querySelector('#brain-pane').getBoundingClientRect().toJSON(),curves:document.querySelector('#curves-pane').getBoundingClientRect().toJSON()}));
  assert.ok(narrow.width<=narrow.viewport+1 && narrow.curves.y>=narrow.brain.bottom-1);
  await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
  await page.locator('#clear-cache').click();await ready();
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  const report={actual_saved_data:true,selected_channel:'ECoG_16',trial_pair:[5,6],pixels,plots,picked_channel:candidate,narrow,exportMessage,errors,external};
  writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
  if(process.argv.includes('--close')){await page.locator('#close-session').click();await page.waitForFunction(()=>document.querySelector('#status-text').textContent==='Session closed');}
  console.log(JSON.stringify(report));
} finally {await browser.close();}
