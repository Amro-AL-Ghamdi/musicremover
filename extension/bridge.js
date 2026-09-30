// Isolated-world content script: relays between inject.js (page world) and the
// background service worker, and pushes settings changes into the page.
(() => {
  const TAG = "__musicremover__";
  // After the extension is reloaded or updated, this copy is orphaned (its
  // chrome.runtime is gone) and background.js injects a fresh one. Step aside.
  const alive = () => { try { return !!chrome.runtime?.id; } catch (_) { return false; } };
  // background.js also injects into tabs that were already loading on install, which can
  // leave two live copies in one tab; both would forward every chunk to the server.
  const other = window.__musicremoverBridgeAlive;
  if (other && other()) return;
  window.__musicremoverBridgeAlive = alive;
  // engine + bleed are passed along so the page can re-process chunks when they change.
  const DEFAULTS = { enabled: true, mode: "wait", chunkSeconds: 60, firstChunkSeconds: 20, readySound: true,
                     engine: "voc_ft_cdx23", bleed: "normal", cache: true };

  const pushSettings = () =>
    chrome.storage.sync.get(DEFAULTS, (settings) => window.postMessage({ [TAG]: "settings", settings }, "*"));

  chrome.storage.onChanged.addListener(pushSettings);
  pushSettings();


  function toBase64(bytes) {
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }
  function fromBase64(b64) {
    const s = atob(b64);
    const out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out.buffer;
  }

  window.addEventListener("message", function onMessage(ev) {
    if (ev.source !== window || !ev.data || !ev.data[TAG]) return;
    if (!alive()) return window.removeEventListener("message", onMessage);
    const m = ev.data;
    if (m[TAG] === "hello") return pushSettings();
    // Cache requests: {type, reqId, ...}; audio travels as base64 to the background.
    if (m[TAG] === "request") {
      if (!["cache-get", "cache-audio", "cache-put"].includes(m.type)) return;
      const msg = { ...m.payload, type: m.type };
      if (msg.audio) { msg.data = toBase64(new Uint8Array(msg.audio)); delete msg.audio; }
      chrome.runtime.sendMessage(msg, (res) => {
        const reply = { [TAG]: "reply", reqId: m.reqId };
        if (chrome.runtime.lastError || !res || res.error) {
          reply.error = chrome.runtime.lastError?.message || res?.error || "no response";
          return window.postMessage(reply, "*");
        }
        reply.res = res;
        if (res.data) {
          reply.audio = fromBase64(res.data);
          delete res.data;
          return window.postMessage(reply, "*", [reply.audio]);
        }
        window.postMessage(reply, "*");
      });
      return;
    }
    if (m[TAG] !== "separate") return;
    chrome.runtime.sendMessage({ type: "separate", mime: m.mime, data: toBase64(m.bytes) }, (res) => {
      const reply = { [TAG]: "result", id: m.id };
      if (chrome.runtime.lastError || !res) reply.error = chrome.runtime.lastError?.message || "no response";
      else if (res.error) reply.error = res.error;
      if (reply.error) return window.postMessage(reply, "*");
      reply.audio = fromBase64(res.data);
      window.postMessage(reply, "*", [reply.audio]);
    });
  });
})();
