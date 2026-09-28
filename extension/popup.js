const DEFAULTS = { enabled: true, mode: "wait", chunkSeconds: 60, firstChunkSeconds: 20, server: "http://127.0.0.1:8765" };
const $ = (id) => document.getElementById(id);

chrome.storage.sync.get(DEFAULTS, (s) => {
  for (const k of Object.keys(DEFAULTS)) {
    const el = $(k);
    if (el.type === "checkbox") el.checked = s[k];
    else el.value = s[k];
    el.addEventListener("change", () => {
      let v = el.type === "checkbox" ? el.checked : el.type === "number" ? Number(el.value) : el.value;
      chrome.storage.sync.set({ [k]: v }, k === "server" ? checkServer : undefined);
    });
  }
  checkServer();
});

function checkServer() {
  const st = $("status");
  chrome.runtime.sendMessage({ type: "health" }, (res) => {
    if (chrome.runtime.lastError || !res || res.error) {
      st.className = "bad";
      st.textContent = "Server not reachable – run: python server/server.py";
    } else {
      st.className = "ok";
      st.textContent = `Server OK – ${res.model} on ${res.device}`;
    }
  });
}
