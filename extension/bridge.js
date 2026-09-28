// Isolated-world content script: relays between inject.js (page world) and the
// background service worker, and pushes settings changes into the page.
(() => {
  const TAG = "__musicremover__";
  // engine + bleed are passed along so the page can re-process chunks when they change.
  const DEFAULTS = { enabled: true, mode: "wait", chunkSeconds: 60, firstChunkSeconds: 20, readySound: true,
                     engine: "bandit", bleed: "normal" };

  const pushSettings = () =>
    chrome.storage.sync.get(DEFAULTS, (settings) => window.postMessage({ [TAG]: "settings", settings }, "*"));

  chrome.storage.onChanged.addListener(pushSettings);
  pushSettings();

  // After the extension is reloaded or updated, this copy is orphaned (its
  // chrome.runtime is gone) and background.js injects a fresh one. Step aside.
  const alive = () => { try { return !!chrome.runtime?.id; } catch (_) { return false; } };

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
