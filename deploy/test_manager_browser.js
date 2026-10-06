// Hermetic Brave UI + WebSocket integration. Bot processes and rooms are simulated.
const assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {chromium}=require('playwright'),WebSocket=require('ws');
const {startServer}=require('./hybrid_server'),{BotManager}=require('./bot_manager'),{FakeChild}=require('./test_bot_manager');
const fixtures=require('./test_hybrid');
async function run(){
  const children=[],manager=new BotManager({model:path.join(__dirname,'rs4_public/model.onnx'),spawn:(_file,args)=>{
    const child=new FakeChild();children.push(child);const id=children.length-1,host=!args.includes('--join');
    setTimeout(()=>{if(!child.connected)return;child.emit('message',{type:'connected',playerId:id,isHost:host});if(host)child.emit('message',{type:'link',link:'https://www.haxball.com/play?c=fixture123'});child.emit('message',{type:'state',team:0,playing:false});},40);return child;
  }});
  const service=await startServer({port:0,token:'manager-browser-token-32-characters',manager,loader:async()=>({meta:fixtures.meta,session:{run:async()=>({logits:{data:new Float32Array(18)}})},name:'fixture.onnx',hash:'fixture'}),geometry:()=>fixtures.geometry});
  let context;
  try{
    const rejected=new WebSocket(`ws://127.0.0.1:${service.port}/fleet`,{origin:'chrome-extension://'+'a'.repeat(32)});
    rejected.on('open',()=>rejected.send(JSON.stringify({type:'auth',token:'wrong'})));
    await new Promise((resolve,reject)=>{rejected.on('close',code=>{try{assert.equal(code,1008);resolve();}catch(e){reject(e);}});rejected.on('error',reject);});
    assert.equal(children.length,0,'unauthenticated request cannot start bots');
    const extension=path.join(__dirname,'hybrid_extension'),profile=fs.mkdtempSync(path.join(os.tmpdir(),'rs4-manager-test-'));
    context=await chromium.launchPersistentContext(profile,{executablePath:'C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe',headless:true,ignoreDefaultArgs:['--disable-extensions'],args:[`--disable-extensions-except=${extension}`,`--load-extension=${extension}`]});
    const worker=context.serviceWorkers()[0]||await context.waitForEvent('serviceworker');
    const extensionId=new URL(worker.url()).host,page=await context.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`chrome-extension://${extensionId}/options.html`);
    await page.evaluate(async settings=>chrome.storage.local.set(settings),{port:service.port,token:service.token});
    await page.goto(`chrome-extension://${extensionId}/manager.html`);
    await page.getByText('Gestor disponible',{exact:true}).waitFor();
    await page.getByLabel('Nombre del bot 1',{exact:true}).fill('Arquero');
    await page.getByLabel('Avatar del bot 1',{exact:true}).fill('GK');
    await page.getByRole('button',{name:'Agregar otro bot',exact:true}).click();
    await page.getByLabel('Nombre del bot 2',{exact:true}).fill('Delantero');
    await page.getByLabel('Equipo del bot 2',{exact:true}).selectOption('2');
    await page.locator('#mode').selectOption('host');
    await page.locator('#headless-token').fill('fake-headless-secret');
    await page.locator('#room-password').fill('private-room');
    // Reusable presets survive reload and never launch bots or store credentials.
    await page.locator('#preset-name').fill('Práctica');await page.locator('#preset-save').click();
    await page.waitForFunction(()=>document.getElementById('preset-status').textContent.includes('guardado en este navegador'));
    const presetA=await page.locator('#preset-select').inputValue();
    let presetStore=await page.evaluate(()=>chrome.storage.local.get(null));
    assert(!JSON.stringify(presetStore).includes('fake-headless-secret'));assert(!JSON.stringify(presetStore).includes('private-room'));
    await page.locator('#room-name').fill('Cancha de los viernes');
    assert.equal(await page.locator('#preset-dirty').textContent(),'Cambios sin guardar');
    await page.locator('#preset-save').click();
    await page.waitForFunction(async id=>(await chrome.storage.local.get('rs4Preset:'+id))['rs4Preset:'+id].config.name==='Cancha de los viernes',presetA);
    await page.locator('#preset-name').fill('Amigos');await page.locator('#preset-copy').click();
    await page.waitForFunction(id=>document.getElementById('preset-select').value!==id,presetA);
    const presetB=await page.locator('#preset-select').inputValue();
    await page.locator('#preset-delete').click();await page.waitForFunction(()=>document.getElementById('preset-status').textContent.includes('eliminado'));
    await page.locator('#preset-undo').click();await page.waitForFunction(()=>document.getElementById('preset-status').textContent.includes('restaurado'));
    assert.equal(await page.locator('#preset-select').inputValue(),presetB);
    await page.locator('#room-name').fill('Cambio sin guardar');
    await page.locator('#preset-select').selectOption(presetA);await page.locator('#preset-load').click();
    await page.waitForFunction(()=>document.getElementById('room-name').value==='Cancha de los viernes');
    await page.locator('#preset-undo').click();await page.waitForFunction(()=>document.getElementById('room-name').value==='Cambio sin guardar');
    await page.locator('#preset-select').selectOption(presetA);await page.locator('#preset-load').click();
    await page.waitForFunction(()=>document.getElementById('room-name').value==='Cancha de los viernes');
    await page.reload();await page.getByText('Gestor disponible',{exact:true}).waitFor();
    assert.equal(await page.locator('#preset-name').inputValue(),'Práctica');assert.equal(await page.locator('#profiles .profile').count(),2);
    assert.equal(await page.locator('#mode').inputValue(),'host');assert.equal(await page.locator('#headless-token').inputValue(),'');assert.equal(await page.locator('#room-password').inputValue(),'');
    assert.equal(children.length,0,'loading/restoring presets cannot start connections');
    // Offline edits retain model choices; reconnect must not reset the form.
    await page.evaluate(()=>chrome.storage.local.remove('token'));await page.reload();
    await page.waitForFunction(()=>document.getElementById('connection-status').textContent.includes('Guardá puerto'));
    assert.equal(await page.getByLabel('Modelo del bot 1',{exact:true}).inputValue(),'current');
    await page.locator('#preset-name').fill('Sin conexión');await page.locator('#preset-copy').click();
    await page.waitForFunction(()=>document.getElementById('preset-status').textContent.includes('«Sin conexión» guardado'));
    await page.evaluate(async ({token,id})=>chrome.storage.local.set({token,rs4LastPreset:id}),{token:service.token,id:presetA});
    // A saved model absent from the service remains visible and blocks launch.
    await page.evaluate(async id=>{const key='rs4Preset:'+id,p=(await chrome.storage.local.get(key))[key];p.config.profiles[0].model='unavailable-fixture';await chrome.storage.local.set({[key]:p});},presetA);
    await page.reload();await page.getByText('Gestor disponible',{exact:true}).waitFor();
    await page.waitForFunction(()=>document.getElementById('launch-help').textContent.includes('no está disponible'));
    assert.equal(await page.getByLabel('Modelo del bot 1',{exact:true}).inputValue(),'unavailable-fixture');assert(await page.locator('#start').isDisabled());
    await page.getByLabel('Modelo del bot 1',{exact:true}).selectOption('current');await page.locator('#preset-save').click();
    await page.waitForFunction(()=>document.getElementById('preset-status').textContent.includes('guardado en este navegador'));
    assert(await page.locator('#start').isEnabled());
    // A storage failure must leave the saved preset intact and the edit recoverable.
    await page.locator('#room-name').fill('Edición recuperable');
    await page.evaluate(()=>{globalThis.originalSet=chrome.storage.local.set;chrome.storage.local.set=async()=>{throw Error('test storage unavailable');};});
    await page.locator('#preset-save').click();
    await page.waitForFunction(()=>document.getElementById('preset-status').textContent.includes('No se pudo guardar'));
    assert.equal(await page.locator('#room-name').inputValue(),'Edición recuperable');
    assert.equal(await page.evaluate(async id=>(await chrome.storage.local.get('rs4Preset:'+id))['rs4Preset:'+id].config.name,presetA),'Cancha de los viernes');
    await page.evaluate(()=>{chrome.storage.local.set=globalThis.originalSet;});
    await page.locator('#preset-load').click();
    await page.waitForFunction(()=>document.getElementById('room-name').value==='Cancha de los viernes');

    await page.locator('#headless-token').fill('fake-headless-secret');await page.locator('#room-password').fill('private-room');
    await page.getByRole('button',{name:'Crear sala e iniciar bots',exact:true}).click();
    await page.waitForFunction(()=>document.querySelectorAll('.instance').length===2&&[...document.querySelectorAll('.badge')].every(e=>e.textContent==='Conectado'));
    assert.equal(await page.locator('#active-count').textContent(),'2');
    assert.equal(children.length,2);assert.equal(await page.locator('#headless-token').inputValue(),'');assert.equal(await page.locator('#room-password').inputValue(),'');
    const stored=await page.evaluate(()=>chrome.storage.local.get(null));assert(!JSON.stringify(stored).includes('fake-headless-secret'));assert(!JSON.stringify(stored).includes('private-room'));
    await page.getByRole('button',{name:'Pausar bot',exact:true}).first().click();await page.getByRole('button',{name:'Reanudar bot',exact:true}).waitFor();
    await page.getByRole('button',{name:'Iniciar partido',exact:true}).click();assert(children[0].sent.some(m=>m.type==='game'&&m.action==='start'));
    await page.screenshot({path:path.resolve(__dirname,'../reports/rs4_presets_manager.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile layout fits viewport');
    await page.screenshot({path:path.resolve(__dirname,'../reports/rs4_presets_manager-mobile.png'),fullPage:true});
    await page.getByRole('button',{name:'Detener todos',exact:true}).click();await page.waitForFunction(()=>[...document.querySelectorAll('.badge')].every(e=>e.textContent==='Detenido'));
    assert.equal(await page.locator('#active-count').textContent(),'0');
    await page.locator('#mode').selectOption('join');await page.locator('#room').fill('https://www.haxball.com/play?c=fixture456');
    await page.getByRole('button',{name:'Agregar bots a la sala',exact:true}).click();await page.waitForFunction(()=>document.querySelectorAll('.instance').length===4);
    await page.close();
    await new Promise(r=>setTimeout(r,1000));assert.equal(manager.snapshot().bots.filter(b=>b.status==='connected').length,2,'closing console keeps requested bots active');
    assert.deepEqual(errors,[]);
    const report={passed:true,browser:'brave',realRoom:false,simulatedProcesses:true,hostAndJoin:true,multipleProfiles:true,avatar:true,teamCommands:true,pauseResume:true,stopAll:true,rejectInvalidToken:true,secretsNotStored:true,desktopAndMobile:true,popupCloseKeepsBots:true,presetCreateUpdateCopy:true,presetDeleteUndo:true,presetLoadUndo:true,presetRestore:true,presetOffline:true,presetMissingModelBlocked:true,presetSecretsExcluded:true,presetStorageFailureRecoverable:true};
    fs.writeFileSync(path.join(__dirname,'../reports/rs4_presets_manager.json'),JSON.stringify(report,null,2));console.log(report);
  }finally{await context?.close();await service.close();assert(children.every(c=>!c.connected),'service shutdown stops all bots');}
}
run().catch(e=>{console.error(e);process.exitCode=1;});
