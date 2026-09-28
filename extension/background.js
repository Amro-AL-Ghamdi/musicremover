// Service worker: forwards audio chunks to the local separation server.
// (Done here rather than from the page so YouTube's CSP and local-network
// restrictions don't apply; host_permissions grant access to localhost.)
const DEFAULT_SERVER = "http://127.0.0.1:8765";

async function serverUrl() {
  const { server } = await chrome.storage.sync.get({ server: DEFAULT_SERVER });
  return server.replace(/\/+$/, "");
}

// Bleed-suppression strength per model and level, chosen from bench/README.md:
// Demucs keeps improving up to much higher strengths before speech suffers.
const STRENGTHS = {
  bandit: { off: 0, normal: 1, strong: 4 },
  demucs: { off: 0, normal: 16, strong: 64 },
  dnr_demucs: { off: 0, normal: 1, strong: 4 },
};

// engine: model; device: "auto" (GPU if the server has one) or "cpu"; bleed: off|normal|strong.
async function processingOptions() {
  const o = await chrome.storage.sync.get({ engine: "bandit", device: "auto", bleed: "normal" });
  const levels = STRENGTHS[o.engine] || STRENGTHS.bandit;
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

async function health() {
  const res = await fetch(`${await serverUrl()}/health`);
  return res.json();
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const job = msg.type === "separate" ? separate(msg) : msg.type === "health" ? health() : null;
  if (!job) return false;
  job.then(sendResponse, (e) => sendResponse({ error: String(e.message || e) }));
  return true; // async response
});
