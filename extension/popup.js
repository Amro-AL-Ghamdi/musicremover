const DEFAULTS = {
  enabled: true, mode: "wait", readySound: true, engine: "bandit", bleed: "normal", device: "auto",
  chunkSeconds: 60, firstChunkSeconds: 20, server: "http://127.0.0.1:8765",
};
const ENGINE_HINTS = {
  bandit: "Removes instruments, keeps dialogue and sound effects. Recommended.",
  demucs: "About 6× faster, but sound effects are removed together with the music.",
  dnr_demucs: "Keeps dialogue and effects like BandIt. Untested model: first use downloads it from Zenodo.",
  voc_ft: "UVR-MDX-NET-Voc_FT. Fast, with less music under speech than Demucs; sound effects are removed with the music.",
  melband: "MelBand RoFormer (Kim, fine-tuned by unwa). Least music of all models, but about 5× slower than Voc FT. Removes sound effects.",
  voc_ft_dnr: "Voices from Voc FT plus sound effects from DnR Demucs. Runs both models (about 1.4× Voc FT alone). Untested: first use downloads DnR Demucs from Zenodo.",
};
// Numbers from bench/README.md.
const HINTS = {
  off: "Model output as is. Faint music can remain in quiet pauses.",
  normal: "Recommended. Removes leftover music in pauses and lowers it under speech, at almost no cost.",
  strong: "Removes a little more leftover music; speech and effects that overlap music get slightly softer.",
};
const $ = (id) => document.getElementById(id);

chrome.storage.sync.get(DEFAULTS, (s) => {
  for (const k of Object.keys(DEFAULTS)) {
    const el = $(k);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = s[k];
    else el.value = s[k];
    el.addEventListener("change", () => {
      const v = el.type === "checkbox" ? el.checked
        : el.type === "number" ? Number(el.value) : el.value;
      chrome.storage.sync.set({ [k]: v }, k === "server" ? checkServer : undefined);
      if (k === "bleed") $("bleedHint").textContent = HINTS[v];
      if (k === "engine") $("engineHint").textContent = ENGINE_HINTS[v];
    });
  }
  // "device" is stored as auto|cpu and shown as the Force CPU checkbox.
  $("forceCpu").checked = s.device === "cpu";
  $("forceCpu").addEventListener("change", () => {
    chrome.storage.sync.set({ device: $("forceCpu").checked ? "cpu" : "auto" }, checkServer);
  });
  $("bleedHint").textContent = HINTS[s.bleed];
  $("engineHint").textContent = ENGINE_HINTS[s.engine];
  checkServer();
});

function checkServer() {
  const st = $("gpu");
  chrome.runtime.sendMessage({ type: "health" }, (res) => {
    if (chrome.runtime.lastError || !res || res.error) {
      st.className = "status bad";
      st.textContent = "Server not reachable. Start it with: python server/server.py";
      return;
    }
    const forceCpu = $("forceCpu").checked;
    if (res.forced_device) {
      st.className = "status warn";
      st.textContent = `Device fixed by server (MR_DEVICE=${res.forced_device})`;
    } else if (!res.gpu) {
      st.className = "status warn";
      // e.g. an AMD/NVIDIA card is present but PyTorch was installed without ROCm/CUDA.
      st.textContent = res.gpu_hint || "No GPU found: running on CPU (slow)";
    } else if (forceCpu) {
      st.className = "status warn";
      st.textContent = `GPU found (${res.gpu}), but CPU is forced`;
    } else {
      st.className = "status ok";
      st.textContent = `Using GPU: ${res.gpu}`;
    }
  });
}
