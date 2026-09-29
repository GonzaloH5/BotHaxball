// Diagnóstico de conexión: intenta unirse a una sala y registra toda la negociación WebRTC
// (candidatos ICE locales/remotos, estados ICE y del datachannel) para ver dónde se corta.
//
// Uso: node bridge/diag_join.js <link o id de sala> [--password x] [--stun stun:host:port] [--sin-arreglo]

const polyfill = require("node-datachannel/polyfill");
const nodeDataChannel = require("node-datachannel");

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
const target = argv.find((a, i) => !a.startsWith("--") && !(i > 0 && argv[i - 1].startsWith("--")));
const m = target.match(/[?&]c=([\w-]+)/);
const roomId = m ? m[1] : target;

if (argv.includes("--verbose")) nodeDataChannel.initLogger("Debug");

const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(2)}s]`, ...a);

function candType(c) {
  const mm = / typ (\w+)/.exec(c || "");
  const addr = (c || "").split(" ").slice(4, 6).join(":");
  const proto = (c || "").split(" ")[2];
  return `${mm ? mm[1] : "?"} ${proto} ${addr}`;
}

const FixedPC = require("./haxball").RTCPeerConnection;
const PC_BASE = argv.includes("--sin-arreglo") ? polyfill.RTCPeerConnection : FixedPC;
class LoggedPC extends PC_BASE {
  constructor(config, ...rest) {
    log("RTCPeerConnection config:", JSON.stringify(config));
    super(config, ...rest);
    if (argv.includes("--sin-listeners-pc")) return; // bisección
    this.addEventListener("icecandidate", (e) => log("  local cand :", e.candidate ? candType(e.candidate.candidate) : "(fin)"));
    this.addEventListener("iceconnectionstatechange", () => log("  ICE:", this.iceConnectionState));
    this.addEventListener("icegatheringstatechange", () => log("  gathering:", this.iceGatheringState));
    this.addEventListener("connectionstatechange", () => log("  conexión:", this.connectionState));
    this.addEventListener("datachannel", (e) => log("  datachannel remoto:", e.channel.label));
  }
  setRemoteDescription(desc, ...cbs) {
    const cands = (desc.sdp.match(/a=candidate:[^\r\n]+/g) || []).map((c) => candType(c.slice(2)));
    log(`  SDP remoto (${desc.type}), candidatos en SDP: ${cands.length ? cands.join(" | ") : "ninguno"}`);
    return super.setRemoteDescription(desc, ...cbs);
  }
  async addIceCandidate(c) {
    log("  cand remoto:", c && c.candidate ? candType(c.candidate) : "(fin)");
    return super.addIceCandidate(c);
  }
  createDataChannel(label, opts) {
    const ch = super.createDataChannel(label, opts);
    log(`  datachannel local "${label}"`, JSON.stringify(opts || {}));
    if (argv.includes("--canales-sin-log")) return ch; // para aislar si los listeners cambian algo
    ch.addEventListener("open", () => log(`  datachannel "${label}" abierto`));
    ch.addEventListener("close", () => log(`  datachannel "${label}" cerrado`));
    ch.addEventListener("error", (e) => log(`  datachannel "${label}" error`, e && e.message));
    const send = ch.send.bind(ch);
    let nOut = 0;
    ch.send = (data) => {
      nOut++;
      const len = data.byteLength ?? data.length;
      if (nOut <= 5) log(`  → envío "${label}" #${nOut} (${len} bytes, estado canal ${ch.readyState}, buffer ${ch.bufferedAmount})`);
      try { return send(data); } catch (e) { log(`  → ERROR al enviar por "${label}":`, e && e.message); throw e; }
    };
    let n = 0, bytes = 0;
    ch.addEventListener("message", (e) => {
      n++; bytes += e.data.byteLength || e.data.length || 0;
      if (n <= 3 || n % 200 === 0) log(`  datachannel "${label}" mensaje #${n} (${bytes} bytes acumulados)`);
    });
    return ch;
  }
}

const haxball = require("./haxball");
// --pc-simple: usa la conexión de haxball.js sin la capa de logging (igual que record.js)
const API = haxball(opt("--stun", null) ? { stunServer: opt("--stun") } : undefined,
  argv.includes("--pc-simple") ? {} : { PeerConnection: LoggedPC });
const { Utils, Room } = API;

(async () => {
  log("interfaz fijada (bindAddress):", (await API.ready) || "ninguna (todas)");
  // --como-grabador: mismo nombre/avatar/identidad guardada que record.js (para aislar diferencias)
  const likeRec = argv.includes("--como-grabador");
  const fs = require("fs");
  const storedKey = likeRec ? fs.readFileSync(require("path").join(__dirname, ".auth_key"), "utf8").trim() : null;
  const authObj = likeRec ? await Utils.authFromKey(storedKey) : (await Utils.generateAuth())[1];
  const storage = likeRec
    ? { player_name: "[BOT] rec", avatar: "🤖", player_auth_key: storedKey, crappy_router: true }
    : { player_name: "[BOT] diag", avatar: "🔧", crappy_router: true };
  log(`uniéndome a ${roomId}...`);
  Room.join({ id: roomId, password: opt("--password", null), authObj }, {
    storage,
    onConnInfo: (state, extra) => log("estado node-haxball:", state, typeof extra === "string" ? extra.slice(0, 80) : ""),
    onOpen: (room) => {
      log(`✅ DENTRO de "${room.name}". Salgo.`);
      setTimeout(() => { room.leave(); process.exit(0); }, 1000);
    },
    onClose: (msg) => {
      let why = ""; try { why = msg ? msg.toString() : ""; } catch (_) {}
      log("❌ cerrado:", why);
      setTimeout(() => process.exit(0), 300);
    },
  });
  setTimeout(() => { log("timeout global"); process.exit(0); }, 40000);
})();
