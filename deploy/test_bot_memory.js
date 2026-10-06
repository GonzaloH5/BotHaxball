// Ejecuta el BotPlugin real con una sala e inferencia simuladas (sin conectarse a HaxBall).
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { PolicyMemory } = require("./policy_memory");

(async () => {
  const code = fs.readFileSync(path.join(__dirname, "bot.js"), "utf8");
  const plugin = code.slice(code.indexOf("function BotPlugin("), code.indexOf("\n(async () =>"));
  const disc = () => ({ pos: {x:0,y:0}, speed: {x:0,y:0} });
  const me = { id:1, team:{id:1}, disc:disc(), isKicking:true };
  const sent = [], pending = [];
  const room = { isHost:true, currentPlayer:me, currentPlayerId:1, currentFrameNo:0,
    state:{players:[me]}, gameState:{state:1,physicsState:{discs:[disc()]}},
    setKeyState: value => sent.push(value) };
  const context = {
    Plugin: function() { this.room = room; }, AllowFlags:{CreateRoom:1,JoinRoom:2},
    PolicyMemory, META:{recurrent:true,memory_size:2,n_actions:18,n_per_team:1,frame_skip:1,max_ticks:7200,goal_x:370},
    UNIVERSAL:false, RS4Z:false, R_PLAYER:15, R_BALL:10, KICK_REACH:4,
    ort:{Tensor:class {constructor(type,data,dims){this.type=type;this.data=data;this.dims=dims;}}},
    buildObs:()=>[0], sample:()=>3, decodeAction:()=>({dirX:1,dirY:0,kick:false}),
    Utils:{keyState:x=>x}, console,
  };
  vm.createContext(context);
  vm.runInContext(plugin,context);
  const bot = new context.BotPlugin({run:feeds=>new Promise(resolve=>pending.push({feeds,resolve}))},true);
  const flush = () => new Promise(resolve => setImmediate(resolve));
  const output = {logits:{data:[0,0,0,1]},memory_out:{data:[0.2,0.4]}};
  bot.onGameTick();
  assert.equal(pending[0].feeds.previous_action.data[0],18n);
  bot.onTeamGoal();
  const resetWrites = sent.length; // soltar teclas (setKeyState(0)) es parte del reset seguro
  pending[0].resolve(output);
  await flush();
  assert.equal(sent.length,resetWrites,"no aplicar decisiones anteriores al gol");
  bot.onGameTick();
  pending[1].resolve(output);
  await flush();
  assert.equal(sent.length,resetWrites+1);
  bot.onGameTick();
  assert.equal(pending[2].feeds.previous_action.data[0],3n);
  assert(Math.abs(pending[2].feeds.memory.data[0]-0.2)<1e-6);
  bot.onPlayerTeamChange();
  const teamResetWrites = sent.length;
  pending[2].resolve(output);
  await flush();
  assert.equal(sent.length,teamResetWrites,"no aplicar decisiones de otro equipo");
  bot.onGameTick();
  assert.equal(pending[3].feeds.previous_action.data[0],18n);
  assert.deepEqual([...pending[3].feeds.memory.data],[0,0]);
  pending[3].resolve(output);
  await flush();
  bot.onGameTick();
  bot.setManagedEnabled(false);
  const pauseWrites=sent.length;
  assert.equal(sent.at(-1),0,'pausing releases movement and kick');
  pending[4].resolve(output);await flush();
  assert.equal(sent.length,pauseWrites,'paused bot discards pending inference');
  bot.onGameTick();assert.equal(pending.length,5,'paused bot does not infer');
  bot.setManagedEnabled(true);bot.onGameTick();
  assert.equal(pending[5].feeds.previous_action.data[0],18n,'resume starts fresh policy memory');
  pending[5].resolve(output);await flush();
  bot.onPlayerChat(99,'!temp 0.7'); // Managed bots do not accept remote chat commands.
  console.log("OK: BotPlugin conserva memoria e invalida inferencias al cambiar de episodio/equipo");
})().catch(e=>{console.error(e);process.exitCode=1;});
