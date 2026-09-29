// node-haxball con arreglos para entrar a salas remotas.
//
// 1) Candidatos ICE del host perdidos. node-haxball llama a setRemoteDescription(answer, onSuccess, onError)
//    con el estilo viejo de callbacks y agrega los candidatos del host dentro de onSuccess. El polyfill de
//    node-datachannel (0.25) sólo implementa la versión con Promise e ignora los callbacks, así que esos
//    candidatos nunca se agregan. Con un host en la misma PC/LAN conecta igual; con uno remoto la conexión
//    muere ("Falló al querer ingresar a la sala").
//
// 2) Carrera al abrir los canales ("Game state timeout"). node-haxball crea 3 datachannels y les pone un
//    onmessage provisorio que descarta; el handler real recién se instala cuando los 3 están abiertos. En un
//    navegador se abren juntos antes de cualquier mensaje; en node-datachannel se abren de a uno y el desafío
//    de autenticación del host (primer mensaje) puede llegar en el medio y perderse: el host espera la
//    respuesta y nunca manda el estado. Se retienen los mensajes hasta que todos los canales de la conexión
//    estén abiertos y luego se entregan en orden al handler vigente.
//
// 3) Varias interfaces de red (VPN tipo Radmin/Hamachi, adaptador de VirtualBox...): se fija la de la ruta por
//    defecto a internet (bindAddress) para no ofrecer rutas inútiles. Forzar con HAXBALL_BIND=<ip local>,
//    desactivar con HAXBALL_BIND=off.
//
// Uso:
//   const API = require("./haxball")();   // en vez de require("node-haxball")()
//   await API.ready;                      // antes de Room.join / Room.create (detecta la interfaz)

const dgram = require("dgram");
const polyfill = require("node-datachannel/polyfill");

let bindAddress = null;
const stats = { heldMessages: 0 }; // mensajes que llegaron antes de abrir todos los canales (arreglo 2)

// IP local que el sistema usa para salir a internet (no manda paquetes: connect() en UDP sólo elige ruta)
function detectBindAddress() {
  const env = process.env.HAXBALL_BIND;
  if (env === "off") return Promise.resolve(null);
  if (env) return Promise.resolve(env);
  return new Promise((resolve) => {
    const s = dgram.createSocket("udp4");
    const done = (ip) => { try { s.close(); } catch (_) {} resolve(ip); };
    s.on("error", () => done(null));
    s.connect(53, "8.8.8.8", () => {
      let ip = null;
      try { ip = s.address().address; } catch (_) {}
      done(ip);
    });
  });
}

class RTCPeerConnection extends polyfill.RTCPeerConnection {
  constructor(config = {}, ...rest) {
    super(bindAddress ? { ...config, bindAddress } : config, ...rest);
    this._hbChannels = [];
  }

  // arreglo 1
  setRemoteDescription(desc, onSuccess, onError) {
    const p = super.setRemoteDescription(desc);
    if (onSuccess || onError) {
      p.then(() => onSuccess && onSuccess(), (e) => onError && onError(e));
    }
    return p;
  }

  // arreglo 2
  createDataChannel(label, opts) {
    const ch = super.createDataChannel(label, opts);
    const channels = this._hbChannels;
    channels.push(ch);
    const allOpen = () => channels.every((c) => c.readyState === "open");
    const queue = [];
    let handler = null;
    // el polyfill llama a `this.onmessage(e)`: interceptamos la propiedad en esta instancia
    Object.defineProperty(ch, "onmessage", {
      configurable: true,
      get: () => (handler ? gate : null),
      set: (h) => { handler = h; },
    });
    const gate = (e) => {
      if (!allOpen()) { stats.heldMessages++; queue.push(e); return; }
      if (handler) handler(e);
    };
    // el listener interno del polyfill (que dispara onopen de node-haxball e instala los handlers reales)
    // se registró antes que éste, así que al llegar acá los handlers ya son los definitivos
    ch.addEventListener("open", () => {
      if (!allOpen()) return;
      for (const c of channels) c.dispatchEvent(new Event("hb-flush"));
    });
    ch.addEventListener("hb-flush", () => {
      while (queue.length && handler) handler(queue.shift());
    });
    return ch;
  }
}

module.exports = function makeAPI(config, { language = true, PeerConnection = RTCPeerConnection } = {}) {
  const API = require("node-haxball")({
    RTCPeerConnection: PeerConnection,
    RTCIceCandidate: polyfill.RTCIceCandidate,
    RTCSessionDescription: polyfill.RTCSessionDescription,
  }, config);
  if (language) {
    // sin un idioma cargado, los errores de conexión llegan sin texto
    const SpanishLanguage = require("node-haxball/examples/languages/spanishLanguage");
    API.Language.current = new SpanishLanguage(API);
  }
  API.ready = detectBindAddress().then((ip) => {
    bindAddress = ip;
    return ip;
  });
  return API;
};

module.exports.RTCPeerConnection = RTCPeerConnection;
module.exports.getBindAddress = () => bindAddress;
module.exports.stats = stats;
