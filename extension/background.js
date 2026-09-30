// Service worker: forwards audio chunks to the local separation server.
// (Done here rather than from the page so YouTube's CSP and local-network
// restrictions don't apply; host_permissions grant access to localhost.)
const DEFAULT_SERVER = "http://127.0.0.1:8765";

async function serverUrl() {
  const { server } = await chrome.storage.sync.get({ server: DEFAULT_SERVER });
  return server.replace(/\/+$/, "");
}

// Bleed-suppression strength per model and level, chosen from bench/README.md:
// Voc FT keeps improving up to much higher strengths before speech suffers.
const STRENGTHS = {
  // Effects filter strength; the voices filter runs at 4x this (engines.VocFTMVSEPEngine).
  voc_ft_cdx23: { off: 0, normal: 4, strong: 16 },
  voc_ft: { off: 0, normal: 16, strong: 64 },
  voc_ft_int8: { off: 0, normal: 16, strong: 64 },
  cdx23: (off: 0, normal: 1, strong: 4 },
};
const DEFAULT_ENGINE = "voc_ft_cdx23";

// engine: model; device: "auto" (GPU if the server has one) or "cpu"; bleed: off|normal|strong.
async function processingOptions() {
  const o = await chrome.storage.sync.get({ engine: DEFAULT_ENGINE, device: "auto", bleed: "normal" });
  if (!STRENGTHS[o.engine]) o.engine = DEFAULT_ENGINE;  // a model that was removed
  const levels = STRENGTHS[o.engine];
  return { ...o, strength: levels[o.bleed] ?? levels.normal };
}

function toBase64(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
function fromBase64(b64) {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

async function separate(msg) {
  const { engine, device, strength } = await processingOptions();
  const q = new URLSearchParams({ mime: msg.mime, engine, device, strength: String(strength) });
  const url = `${await serverUrl()}/separate?${q}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: fromBase64(msg.data),
  });
  if (!res.ok) throw new Error(`server ${res.status}: ${(await res.text()).slice(0, 200)}`);
  return { data: toBase64(new Uint8Array(await res.arrayBuffer())) };
}

// ---------------------------------------------------------------------------
// Cache of processed audio for the last CACHE_VIDEOS videos, so rewatching (or seeking
// back into) a video doesn't process it again. Stored per video and model settings
// ("profile"), as time ranges: chunk boundaries differ between viewings, so the page
// reuses whatever ranges are stored and only sends the gaps. Kept in the extension's
// own IndexedDB (not YouTube's), roughly 1 MB per minute of audio.
// ---------------------------------------------------------------------------

const CACHE_VIDEOS = 10;
let dbPromise = null;

function db() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open("musicremover-cache", 1);
      req.onupgradeneeded = () => {
        const d = req.result;
        d.createObjectStore("videos", { keyPath: "videoId" }); // {videoId, duration, lastUsed}
        const chunks = d.createObjectStore("chunks", { keyPath: "id", autoIncrement: true });
        chunks.createIndex("video", "videoId"); // {videoId, profile, start, end, audio}
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => { dbPromise = null; reject(req.error); };
    });
  }
  return dbPromise;
}

// Runs fn(stores) in one transaction; resolves with fn's result once it has committed.
async function tx(mode, fn) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction(["videos", "chunks"], mode);
    let result;
    t.oncomplete = () => resolve(result);
    t.onerror = t.onabort = () => reject(t.error);
    Promise.resolve(fn({ videos: t.objectStore("videos"), chunks: t.objectStore("chunks") }))
      .then((r) => { result = r; }, (e) => { try { t.abort(); } catch (_) {} reject(e); });
  });
}

const req = (r) => new Promise((resolve, reject) => { r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); });

function deleteVideo(st, videoId) {
  st.videos.delete(videoId);
  const c = st.chunks.index("video").openKeyCursor(IDBKeyRange.only(videoId));
  c.onsuccess = () => { const cur = c.result; if (cur) { st.chunks.delete(cur.primaryKey); cur.continue(); } };
}

// A video whose length doesn't match what's stored isn't the same audio (e.g. the
// uploader replaced it); drop what's stored for it.
const sameLength = (v, duration) => !v.duration || !duration || Math.abs(v.duration - duration) < 2;

async function cacheGet({ videoId, profile, duration }) {
  return tx("readwrite", async (st) => {
    const v = await req(st.videos.get(videoId));
    if (!v) return { chunks: [] };
    if (!sameLength(v, duration)) { deleteVideo(st, videoId); return { chunks: [] }; }
    st.videos.put({ ...v, lastUsed: Date.now() });
    const all = await req(st.chunks.index("video").getAll(IDBKeyRange.only(videoId)));
    return { chunks: all.filter((c) => c.profile === profile).map((c) => ({ key: c.id, start: c.start, end: c.end })) };
  });
}

async function cacheAudio({ key }) {
  const c = await tx("readonly", (st) => req(st.chunks.get(key)));
  if (!c) throw new Error("not cached");
  return { data: toBase64(new Uint8Array(c.audio)) };
}

async function cachePut({ videoId, profile, duration, start, end, data }) {
  await tx("readwrite", async (st) => {
    const v = await req(st.videos.get(videoId));
    if (v && !sameLength(v, duration)) deleteVideo(st, videoId);
    st.videos.put({ videoId, duration, lastUsed: Date.now() });
    // A new chunk replaces stored ones it contains (same stretch cut differently).
    const all = await req(st.chunks.index("video").getAll(IDBKeyRange.only(videoId)));
    for (const c of all) {
      if (c.profile === profile && c.start >= start - 0.05 && c.end <= end + 0.05) st.chunks.delete(c.id);
    }
    st.chunks.add({ videoId, profile, start, end, audio: fromBase64(data).buffer });
    // Keep only the most recently watched videos.
    const videos = await req(st.videos.getAll());
    videos.sort((a, b) => b.lastUsed - a.lastUsed);
    for (const old of videos.slice(CACHE_VIDEOS)) if (old.videoId !== videoId) deleteVideo(st, old.videoId);
  });
  return {};
}

async function cacheStats() {
  return tx("readonly", async (st) => {
    const videos = await req(st.videos.count());
    let bytes = 0;
    await new Promise((resolve, reject) => {
      const c = st.chunks.openCursor();
      c.onsuccess = () => { const cur = c.result; if (!cur) return resolve(); bytes += cur.value.audio.byteLength; cur.continue(); };
      c.onerror = () => reject(c.error);
    });
    return { videos, bytes };
  });
}

async function cacheClear() {
  await tx("readwrite", (st) => { st.videos.clear(); st.chunks.clear(); });
  return {};
}

const HANDLERS = {
  separate: (m) => separate(m),
  health: () => health(),
  "cache-get": cacheGet,
  "cache-audio": cacheAudio,
  "cache-put": cachePut,
  "cache-stats": cacheStats,
  "cache-clear": cacheClear,
};

async function health() {
  const res = await fetch(`${await serverUrl()}/health`);
  return res.json();
}

// Content scripts from the manifest only reach pages loaded after install. Inject into
// YouTube tabs that are already open, so they work without a page reload. inject.js
// takes over from any older copy of itself and restarts the playing video so its
// audio goes through the new hooks.
async function injectIntoOpenTabs() {
  const tabs = await chrome.tabs.query({ url: ["https://www.youtube.com/*", "https://m.youtube.com/*"] });
  for (const tab of tabs) {
    const target = { tabId: tab.id };
    try {
      await chrome.scripting.executeScript({ target, files: ["bridge.js"] });
      await chrome.scripting.executeScript({ target, files: ["inject.js"], world: "MAIN" });
    } catch (e) {
      console.warn("[musicremover] couldn't inject into tab", tab.id, e);
    }
  }
}

// Settings saved with a model that has since been removed switch to the default.
async function migrateEngine() {
  const { engine } = await chrome.storage.sync.get("engine");
  if (engine && !STRENGTHS[engine]) await chrome.storage.sync.set({ engine: DEFAULT_ENGINE });
}

chrome.runtime.onInstalled.addListener(async () => {
  await migrateEngine().catch(() => {});
  injectIntoOpenTabs();
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const handler = HANDLERS[msg.type];
  if (!handler) return false;
  const job = handler(msg);
  job.then(sendResponse, (e) => sendResponse({ error: String(e.message || e) }));
  return true; // async response
});
