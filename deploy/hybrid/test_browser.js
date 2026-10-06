// Hermetic browser test. All HaxBall requests are fulfilled offline; no room connection.
const fs=require('fs'),os=require('os'),path=require('path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
const {officialSource,extractClass}=require('./probe_official');
const {startServer}=require('../hybrid_server');
const fixtures=require('../test_hybrid');
function fixtureHtml(source){
  const native=extractClass(source,'P')+';'+extractClass(source,'ta')+';'+extractClass(source,'bc');
  return `<!doctype html><html><body style="background:#203021;color:white;margin:0"><canvas id="pitch" width="900" height="420" style="max-width:100%;height:auto"></canvas><input id="chat" placeholder="Chat oficial de prueba"><script>
    const M=(self,fn)=>fn.bind(self),Ja=class{},D={h:(fn,...args)=>fn?.(...args)};
    const bindings={KeyW:'Up',KeyA:'Left',KeyS:'Down',KeyD:'Right',ArrowUp:'Up',ArrowDown:'Down',ArrowLeft:'Left',ArrowRight:'Right',KeyX:'Kick',Space:'Kick'};
    const m={j:{Jd:{v:()=>({v:code=>bindings[code]})}}};${native}
    const catalog=${JSON.stringify(fixtures.stadium)};
    const team=id=>({ba:id});
    function nativeDisc(pos,radius=15,color=0xFFFFFF){const d=new ta;d.a.x=pos[0];d.a.y=pos[1];d.V=radius;d.S=color;return d;}
    const discs=catalog.discs.map(d=>nativeDisc(d.pos||[0,0],d.radius??10));
    discs[0]=nativeDisc([0,0],10);
    const players=Array.from({length:8},(_,i)=>({Z:i+1,I:nativeDisc([i*40-160,i*5]),fa:team(i<4?1:2),W:0,Yb:false,Ud:false}));
    discs.push(...players.map(p=>p.I));
    const room={K:players,U:{D:catalog.name,rb:catalog.joints.map(j=>({he:j.d0,ie:j.d1,S:parseInt(j.color,16)})),us:()=>catalog},
      M:{va:{H:discs},Cb:1,Tb:0,Ob:0,Nc:0,Ta:0}};
    const clockStart=performance.now();
    const client={T:room,yc:1,Y:0,connection:{identity:'one-fixture-connection'},A(){this.Y=Math.floor((performance.now()-clockStart)*60/1000);if(room.M)room.M.Nc=this.Y/60;},ta(op){players[0].W=op.input;players[0].Yb=!!(op.input&16);}};
    class FixtureView{
      constructor(){this.za=client;this.W=new bc;
        document.addEventListener('keydown',M(this,this.Fa));document.addEventListener('keyup',M(this,this.ld));
        this.W.Bg=op=>client.ta(op);requestAnimationFrame(M(this,this.sf));}
      Fa(event){if(event.code==='Enter'){document.getElementById('chat').focus();return;}if(event.code==='Escape')return;this.W.Fa(event);}
      ld(event){this.W.ld(event);}
      sf(){if(window.fixture?.pauseFrames)return;this.za.A();this.W.A();requestAnimationFrame(M(this,this.sf));}
      la(){}
    }
    const view=new FixtureView;window.fixture={view,client,room,joinCalls:0,identity:client.connection};
    const ctx=document.getElementById('pitch').getContext('2d');ctx.fillStyle='#709557';ctx.fillRect(10,10,880,400);ctx.strokeStyle='white';ctx.strokeRect(50,40,800,340);ctx.beginPath();ctx.moveTo(450,40);ctx.lineTo(450,380);ctx.stroke();
    for(let i=0;i<8;i++){ctx.beginPath();ctx.fillStyle=i<4?'#ff735d':'#669cff';ctx.arc(280+i*42,180+i*8,9,0,7);ctx.fill();}
  </script></body></html>`;
}
async function run({browser='brave',model,out}={}){
  const executables={brave:'C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe',chrome:'C:/Program Files/Google/Chrome/Application/chrome.exe'};
  const executablePath=executables[browser];if(executablePath&&!fs.existsSync(executablePath))throw Error('Browser not installed: '+browser);
  const {source,url}=await officialSource();
  const service=await startServer({port:0,model,token:'offline-browser-pairing-token-32-characters',
    ...(model?{}:{loader:async()=>({meta:fixtures.meta,session:{run:async()=>({logits:{data:new Float32Array(18)}})},name:'fixture.onnx',hash:'fixture'}),geometry:()=>fixtures.geometry})});
  const profile=fs.mkdtempSync(path.join(os.tmpdir(),'rs4-hybrid-browser-'));
  let context,page,options;
  try{
    const extension=path.resolve(__dirname,'../hybrid_extension');
    context=await chromium.launchPersistentContext(profile,{executablePath,headless:true,ignoreDefaultArgs:['--disable-extensions'],
      args:browser==='chrome'?['--enable-unsafe-extension-debugging']:[`--disable-extensions-except=${extension}`,`--load-extension=${extension}`]});
    let extensionId;
    if(browser==='chrome'){
      // Branded Chrome removed --load-extension. CDP is confined to this hermetic
      // disposable test profile; the shipped extension never requests debugger.
      const cdp=await context.browser().newBrowserCDPSession();extensionId=(await cdp.send('Extensions.loadUnpacked',{path:extension})).id;await cdp.detach();
    }
    if(!extensionId){let worker=context.serviceWorkers()[0];if(!worker)worker=await context.waitForEvent('serviceworker',{timeout:15000});
      extensionId=new URL(worker.url()).host;}
    page=await context.newPage();let unknownClient=false;
    await page.route('**/*',route=>{
      const request=route.request().url();
      if(request===url)return route.fulfill({status:200,contentType:'application/javascript',body:source+(unknownClient?'\n/* unknown client fixture */':'')});
      if(request===url.replace('game-min.js','game.html'))return route.fulfill({status:200,contentType:'text/html',body:fixtureHtml(source)});
      return route.abort();
    });
    const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')console.error('Page console:',m.text());});
    await page.goto(url.replace('game-min.js','game.html'));
    await page.waitForSelector('#rs4-hybrid-panel',{timeout:10000});
    // Manual input must work before credentials, geometry, permission or ONNX.
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='HUMANO');
    await page.keyboard.down('KeyW');await page.keyboard.down('KeyX');await page.waitForFunction(()=>fixture.view.W.ng===17);
    await page.keyboard.up('KeyW');await page.keyboard.up('KeyX');await page.waitForFunction(()=>fixture.view.W.ng===0);
    await page.evaluate(()=>{globalThis.hybridShortcutMessages=[];globalThis.hybridAccepted=0;window.addEventListener('message',event=>{if(event.data?.type==='shortcut')hybridShortcutMessages.push(event.data);if(event.data?.type==='ack'&&event.data.accepted)hybridAccepted++;});});
    options=await context.newPage();await options.goto(`chrome-extension://${extensionId}/options.html`);
    await options.waitForFunction(()=>document.getElementById('port').value.length>0);
    await options.getByRole('button',{name:'Mostrar token',exact:true}).click();
    assert.equal(await options.locator('#token').getAttribute('type'),'text');
    await options.getByRole('button',{name:'Ocultar token',exact:true}).click();
    assert.equal(await options.locator('#token').getAttribute('type'),'password');
    await options.setViewportSize({width:420,height:600});
    assert(await options.locator('#next-step').isHidden(),'game link stays hidden until service connects');
    assert(await options.evaluate(()=>document.querySelector('#connect').getBoundingClientRect().bottom<=innerHeight),'primary connection action fits popup without scrolling');
    assert(await options.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'popup has no horizontal overflow');
    if(out)await options.screenshot({path:path.resolve(out).replace(/\.json$/,'-options.png'),fullPage:true});
    await options.locator('#show-guide').click();
    assert(await options.locator('#copy-command').isVisible(),'startup command is reachable from connection form');
    await options.locator('#startup-guide summary').click();
    await options.locator('#shortcut').focus();await options.keyboard.press('Control+Alt+b');
    assert.equal(await options.locator('#shortcut').inputValue(),'Ctrl + Alt + B');
    await options.getByRole('button',{name:'Guardar atajo',exact:true}).click();
    await options.waitForFunction(()=>document.getElementById('shortcut-status').textContent.includes('Atajo guardado'));
    await options.getByRole('button',{name:'Sin atajo',exact:true}).click();
    await options.waitForFunction(()=>document.getElementById('shortcut-status').textContent.includes('Atajo desactivado'));
    assert.equal(await options.locator('#shortcut').inputValue(),'Sin atajo');
    await options.locator('#shortcut').focus();await options.keyboard.press('Control+Alt+b');
    await options.getByRole('button',{name:'Guardar atajo',exact:true}).click();
    await options.locator('#port').fill(String(service.port));await options.locator('#token').fill(service.token);await options.getByRole('button',{name:'Guardar y conectar',exact:true}).click();
    await options.waitForFunction(()=>document.getElementById('status').textContent.includes('Conectado'),null,{timeout:10000});
    assert.equal(await options.evaluate(async()=>RS4Shortcut.label((await chrome.storage.local.get('shortcut')).shortcut)),'Ctrl + Alt + B');
    assert(await options.locator('#next-step').isVisible(),'connected state offers the next step');
    await options.close();await page.bringToFront();
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('model').textContent.includes('0349dd60'),null,{timeout:10000});
    await page.waitForFunction(()=>!document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('reason').textContent.includes('Esperando'),null,{timeout:15000});
    assert(await page.locator('#pair').isVisible());
    assert(await page.locator('#qualify').isVisible(),'required check is visible with diagnostics collapsed');
    assert(await page.locator('#toggle').isVisible());
    assert.equal(await page.locator('details').getAttribute('open'),null);
    await page.getByText('Opciones y diagnóstico',{exact:true}).click();
    assert.equal(await page.locator('#checklist li').count(),7);
    const downloadEvent=page.waitForEvent('download');
    await page.getByRole('button',{name:'Descargar diagnóstico',exact:true}).click();
    const download=await downloadEvent;
    const exported=JSON.parse(fs.readFileSync(await download.path(),'utf8'));
    assert.equal(exported.version,JSON.parse(fs.readFileSync(path.join(extension,'manifest.json'),'utf8')).version);
    assert.ok(Array.isArray(exported.events));
    assert.ok(!JSON.stringify(exported).includes(service.token),'Diagnostics must exclude pairing credentials');
    assert.ok(!Object.hasOwn(exported,'frameData'),'Diagnostics must exclude gameplay frames');
    await page.getByRole('button',{name:'Limpiar eventos',exact:true}).click();
    assert.equal(await page.locator('#events').textContent(),'Historial limpio.');
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('reason').textContent.includes('casilla de permiso'));
    assert(await page.getByRole('button',{name:'Comprobar configuración',exact:true}).isDisabled());
    await page.keyboard.down('KeyD');await page.keyboard.down('KeyX');await page.waitForFunction(()=>fixture.view.W.ng===24);
    await page.keyboard.up('KeyD');await page.keyboard.up('KeyX');
    await page.evaluate(()=>{fixture.savedStadium=fixture.room.U;fixture.room.U={...fixture.room.U,D:'Sanguchito x4rs',us:()=>({...fixture.savedStadium.us(),name:'Sanguchito x4rs'})};});
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('reason').textContent.includes('Sanguchito x4rs'));
    assert(await page.getByRole('button',{name:'Ceder al bot',exact:true}).isDisabled());
    assert(await page.getByRole('button',{name:'Comprobar configuración',exact:true}).isDisabled());
    await page.keyboard.down('KeyA');await page.keyboard.down('KeyX');await page.waitForFunction(()=>fixture.view.W.ng===20);
    await page.keyboard.up('KeyA');await page.keyboard.up('KeyX');
    await page.evaluate(()=>{fixture.room.U={...fixture.savedStadium,us(){throw Error('Map export disabled');}};});
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('reason').textContent.includes('Map export disabled'));
    await page.keyboard.down('KeyS');await page.keyboard.down('KeyX');await page.waitForFunction(()=>fixture.view.W.ng===18);
    await page.keyboard.up('KeyS');await page.keyboard.up('KeyX');
    await page.evaluate(()=>{fixture.room.U=fixture.savedStadium;});
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('reason').textContent.includes('casilla de permiso'));
    // These are deliberate safety gates, not focus problems. Report each one.
    await page.evaluate(()=>{fixture.room.K[0].fa.ba=0;});
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('reason').textContent.includes('espectador'));
    await page.evaluate(()=>{fixture.room.K[0].fa.ba=1;fixture.savedGame=fixture.room.M;fixture.room.M=null;});
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('reason').textContent.includes('detenido'));
    await page.evaluate(()=>{fixture.room.M=fixture.savedGame;fixture.savedPlayer=fixture.room.K.pop();});
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('reason').textContent.includes('azul 3/4'));
    await page.evaluate(()=>{fixture.room.K.push(fixture.savedPlayer);});
    await page.getByLabel('Esta sala es propia o tengo permiso para usar el bot.').check();
    await page.getByRole('button',{name:'Comprobar configuración',exact:true}).click({timeout:15000});
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('metrics').textContent.includes('Preflight aprobado'),null,{timeout:45000});
    await page.getByText('Opciones y diagnóstico',{exact:true}).click();
    assert(await page.locator('#toggle').isEnabled(),'handoff works without opening diagnostics');
    assert((await page.locator('#reason').textContent()).includes('Configuración lista'),'ready state replaces old preparation messages');
    if(out){
      await page.screenshot({path:path.resolve(out).replace(/\.json$/,'-ready.png'),fullPage:true});
      await page.setViewportSize({width:390,height:844});
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'game panel fits narrow viewport');
      assert(await page.locator('#qualify').isVisible());
      assert(await page.locator('#toggle').isVisible());
      await page.screenshot({path:path.resolve(out).replace(/\.json$/,'-mobile.png'),fullPage:true});
      await page.setViewportSize({width:1280,height:720});
    }
    await page.keyboard.press('ArrowUp');assert.equal(await page.evaluate(()=>fixture.view.W.ng),0,'release before further test');
    await page.evaluate(()=>window.dispatchEvent(new Event('blur')));
    assert(!await page.getByRole('button',{name:'Ceder al bot',exact:true}).isDisabled(),'qualified button can acquire iframe focus on click');
    for(let i=0;i<100;i++){
      await page.getByRole('button',{name:'Ceder al bot',exact:true}).click();
      await page.getByRole('button',{name:'Tomar control',exact:true}).click();
    }
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='HUMANO');
    assert.equal(await page.evaluate(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('key').textContent),'Ctrl + Alt + B');
    await page.keyboard.press('b'); // base key without chosen modifiers must not toggle
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='HUMANO');
    await page.keyboard.down('Control');await page.keyboard.down('Alt');await page.keyboard.down('b');
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='BOT');
    await page.keyboard.down('b'); // auto-repeat must not toggle back
    assert.equal(await page.evaluate(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent),'BOT');
    await page.keyboard.up('b');await page.keyboard.up('Alt');await page.keyboard.up('Control');await page.keyboard.press('Escape');
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='HUMANO');
    assert.equal(await page.evaluate(()=>fixture.client.connection===fixture.identity&&fixture.client.yc===1&&fixture.joinCalls===0),true);
    await page.keyboard.down('KeyX');await page.waitForFunction(()=>fixture.view.W.ng===16);
    await page.keyboard.press('Control+Alt+b');await page.keyboard.press('Escape');await page.waitForFunction(()=>fixture.view.W.ng===0&&document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='HUMANO');
    await page.keyboard.up('KeyX');
    await page.locator('#chat').focus();await page.keyboard.press('Control+Alt+b');
    assert.equal(await page.evaluate(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent),'HUMANO');
    await page.locator('#chat').evaluate(el=>el.blur());await page.getByRole('button',{name:'Ceder al bot',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='BOT');
    // Headless platforms do not consistently emit OS window focus events.
    // Explicitly exercise the page's native blur/focus event route, not a room API.
    const background=await page.evaluate(()=>{fixture.pauseFrames=true;Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});
      window.dispatchEvent(new Event('blur'));document.dispatchEvent(new Event('visibilitychange'));return {frame:fixture.client.Y,accepted:hybridAccepted};});
    await page.waitForFunction(previous=>fixture.client.Y>=previous.frame+90&&hybridAccepted>=previous.accepted+20&&document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='BOT',background,{timeout:7000});
    await page.evaluate(()=>{delete document.hidden;document.dispatchEvent(new Event('visibilitychange'));window.dispatchEvent(new Event('focus'));});
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='BOT');
    await page.keyboard.press('Escape');await page.waitForFunction(()=>fixture.view.W.ng===0&&document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='HUMANO');
    await page.evaluate(()=>{fixture.pauseFrames=false;requestAnimationFrame(()=>fixture.view.sf());});
    // Native chat/controller resets during BOT cannot falsify its last sent mask.
    await page.getByRole('button',{name:'Ceder al bot',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='BOT');
    await page.evaluate(()=>fixture.view.W.Al());
    await page.keyboard.press('Escape');await page.waitForFunction(()=>fixture.view.W.ng===0);
    await page.evaluate(()=>{const bodyChildren=[...document.body.children].filter(el=>el.id!=='rs4-hybrid-panel');document.body.replaceChildren(...bodyChildren);});
    await page.waitForSelector('#rs4-hybrid-panel'); // native boot can replace body
    const other=await context.newPage();await other.route('**/*',route=>{
      if(route.request().url()===url)return route.fulfill({status:200,contentType:'application/javascript',body:source});
      if(route.request().url()===url.replace('game-min.js','game.html'))return route.fulfill({status:200,contentType:'text/html',body:fixtureHtml(source)});
      return route.abort();
    });
    await other.goto(url.replace('game-min.js','game.html'));
    await other.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel')?.shadowRoot.getElementById('reason').textContent.includes('Otra pestaña activa'));
    await other.close();await page.bringToFront();await page.keyboard.press('Escape');
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='HUMANO');
    await page.getByText('Opciones y diagnóstico',{exact:true}).click();
    await page.waitForFunction(()=>!document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('reason').textContent.includes('Tomá control humano'));
    await page.screenshot({path:out?path.resolve(out).replace(/\.json$/,'.png'):path.join(profile,'panel.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    const report={browser,passed:true,extensionId,nativeSourceHash:require('./official_profile').SHA256,handoffs:100,
      samePlayer:true,secondConnection:false,customShortcut:'Ctrl + Alt + B',shortcutRepeatIgnored:true,shortcutDisableWorks:true,chatShortcutIgnored:true,backgroundBot:true,pausedRafWithFreshActions:true,focusEvents:'dispatched/headless',secondTabDenied:true,realRoom:false,
      actionablePrerequisites:true,collapsedActivation:true,popupPrimaryAboveFold:true,responsivePanel:true,manualBeforePairing:true,manualBeforePermission:true,unsupportedMapManualWorks:true,protectedMapManualWorks:true,
      qualification:await page.evaluate(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('metrics').textContent)};
    // Kill only this test's local inference socket; leave native controller running.
    await page.getByRole('button',{name:'Ceder al bot',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='BOT');
    await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});window.dispatchEvent(new Event('blur'));document.dispatchEvent(new Event('visibilitychange'));});
    for(const socket of service.wss.clients)socket.terminate();
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent==='SUSPENDIDO'&&fixture.view.W.ng===0);
    report.serviceFailureNeutral=true;report.backgroundServiceFailureNeutral=true;
    await page.evaluate(()=>{delete document.hidden;document.dispatchEvent(new Event('visibilitychange'));window.dispatchEvent(new Event('focus'));});
    await page.keyboard.down('KeyW');await page.keyboard.down('KeyX');await page.waitForFunction(()=>fixture.view.W.ng===17);
    await page.keyboard.up('KeyW');await page.keyboard.up('KeyX');
    assert.equal(await page.evaluate(()=>document.querySelector('#rs4-hybrid-panel').shadowRoot.getElementById('mode').textContent),'HUMANO');
    report.manualAfterFailure=true;
    unknownClient=true;await page.reload();
    await page.waitForFunction(()=>document.querySelector('#rs4-hybrid-panel')?.shadowRoot.getElementById('reason').textContent.includes('Cliente no compatible'));
    await page.keyboard.down('KeyX');await page.waitForFunction(()=>fixture.view.W.ng===16);await page.keyboard.up('KeyX');
    report.unknownClientManualWorks=true;assert.deepEqual(errors,[]);
    if(out)fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n');return report;
  }catch(error){
    if(options&&!options.isClosed())console.error('Options diagnostic:',await options.evaluate(()=>document.getElementById('status').textContent).catch(()=>null));
    if(page)console.error('Browser diagnostic:',await page.evaluate(()=>({panel:document.querySelector('#rs4-hybrid-panel')?.shadowRoot?.textContent,shortcuts:globalThis.hybridShortcutMessages,focused:document.hasFocus(),frame:globalThis.fixture?.client.Y})).catch(()=>null));
    console.error('Isolated profile:',profile);throw error;
  }finally{await context?.close();await service.close();/* Keep isolated test profile for inspection, never delete broad paths. */}
}
if(require.main===module){const args=process.argv.slice(2),arg=(k,d)=>args.includes(k)?args[args.indexOf(k)+1]:d;
  run({browser:arg('--browser','brave'),model:arg('--model',undefined),out:arg('--out',undefined)}).then(r=>console.log(JSON.stringify(r,null,2))).catch(e=>{console.error(e.stack);process.exitCode=1;});}
module.exports={run,fixtureHtml};
