const DEFAULTS = {
  enabled: true, mode: "wait", readySound: true, engine: "bandit", device: "auto",
  chunkSeconds: 60, firstChunkSeconds: 20, server: "http://127.0.0.1:8765",
};
const HINTS = {
  bandit: "Best balance: removes instruments, keeps dialogue and sound effects. Singing is mostly removed.",
  hybrid: "Like the first, plus sung vocals are put back. About 3× slower.",
  vocals: "Cleanest speech, but sound effects are removed along with the music.",
};
const $ = (id) => document.getElementById(id);

chrome.storage.sync.get(DEFAULTS, (s) => {
  for (const k of Object.keys(DEFAULTS)) {
    const el = $(k);
    if (!el) continue;
    if (el.type === "checkbox") el.checked = s[k];
    else el.value = s[k];
    el.addEventListener("change", () => {
      const v = el.type === "checkbox" ? el.checked : el.type === "number" ? Number(el.value) : el.value;
      chrome.storage.sync.set({ [k]: v }, k === "server" ? checkServer : undefined);
      if (k === "engine") $("engineHint").textContent = HINTS[v];
    });
  }
  // "device" is stored as auto|cpu and shown as the Force CPU checkbox.
  $("forceCpu").checked = s.device === "cpu";
  $("forceCpu").addEventListener("change", () => {
    chrome.storage.sync.set({ device: $("forceCpu").checked ? "cpu" : "auto" }, checkServer);
  });
  $("engineHint").textContent = HINTS[s.engine];
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
