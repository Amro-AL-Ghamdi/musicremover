// Runs in the page's MAIN world at document_start, before YouTube's player.
//
// 1. Capture: hooks MediaSource/SourceBuffer so every audio segment YouTube
//    appends to its buffer (the "browser cache" of the stream) is copied, along
//    with the media time range it covers.
// 2. Chunking: contiguous segments are grouped into ~60 s chunks (init segment
//    + media segments = a standalone, decodable file) and sent via bridge.js to
//    the local server for music removal.
// 3. Playback: the <video>'s own audio is routed through a WebAudio gain node
//    and silenced; the processed (vocals-only) chunks are played in sync with
//    video.currentTime instead.
(() => {
  "use strict";
  const TAG = "__musicremover__";

  const settings = {
    enabled: true,
    chunkSeconds: 60,
    // The first chunk after a (re)start is shorter so playback can begin sooner.
    firstChunkSeconds: 20,
    // What to do when playback reaches audio that isn't processed yet:
    //   "wait"     pause the video until the chunk is ready (never hear music)
    //   "mute"     keep playing, silent
    //   "original" keep playing with the original audio
    mode: "wait",
  };

  const log = (...a) => console.debug("[musicremover]", ...a);

  // ---------------------------------------------------------------------------
  // Container parsing helpers (just enough to find init/media boundaries)
  // ---------------------------------------------------------------------------

  // EBML variable-length int. keepMarker=true for element IDs.
  function readVint(b, pos, keepMarker) {
    const first = b[pos];
    if (first === undefined || first === 0) return null;
    let len = 1;
    while (len <= 8 && !(first & (0x80 >> (len - 1)))) len++;
    if (pos + len > b.length) return null;
    let val = keepMarker ? first : first & (0xff >> len);
    let allOnes = val === (0xff >> len);
    for (let i = 1; i < len; i++) {
      val = val * 256 + b[pos + i];
      if (b[pos + i] !== 0xff) allOnes = false;
    }
    return { val, len, unknown: !keepMarker && allOnes };
  }

  const EBML_ID = 0x1a45dfa3, SEGMENT_ID = 0x18538067, CLUSTER_ID = 0x1f43b675;

  // Returns the offset of the first media data in `b`:
  //   0   -> starts directly with media (cluster / moof)
  //   n   -> bytes [0, n) are an init segment, media follows
  //   len -> the whole buffer is init
  //   -1  -> starts mid-element (continuation of a partial append)
  function splitInit(b, isMp4) {
    if (isMp4) {
      let pos = 0;
      while (pos + 8 <= b.length) {
        const size = ((b[pos] << 24) >>> 0) + (b[pos + 1] << 16) + (b[pos + 2] << 8) + b[pos + 3];
        const type = String.fromCharCode(b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]);
        if (!/^[a-z0-9 ]{4}$/i.test(type)) return -1;
        if (type !== "ftyp" && type !== "moov" && type !== "free" && type !== "skip") return pos;
        if (size < 8) return -1;
        pos += size;
      }
      return pos >= b.length ? b.length : -1;
    }
    const id = readVint(b, 0, true);
    if (!id) return -1;
    if (id.val === CLUSTER_ID) return 0;
    if (id.val !== EBML_ID) return -1;
    // EBML header, then Segment, then Segment children until the first Cluster.
    const hsz = readVint(b, id.len, false);
    if (!hsz) return -1;
    let pos = id.len + hsz.len + hsz.val;
    const seg = readVint(b, pos, true);
    if (!seg || seg.val !== SEGMENT_ID) return -1;
    const ssz = readVint(b, pos + seg.len, false);
    if (!ssz) return -1;
    pos += seg.len + ssz.len;
    while (pos < b.length) {
      const cid = readVint(b, pos, true);
      if (!cid) return b.length;
      if (cid.val === CLUSTER_ID) return pos;
      const csz = readVint(b, pos + cid.len, false);
      if (!csz || csz.unknown) return b.length;
      pos += cid.len + csz.len + csz.val;
    }
    return b.length;
  }

  function toBytes(data) {
    if (data instanceof ArrayBuffer) return new Uint8Array(data.slice(0));
    return new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  }

  function rangesOf(tr) {
    const out = [];
    try { for (let i = 0; i < tr.length; i++) out.push([tr.start(i), tr.end(i)]); } catch (_) {}
    return out;
  }

  // Span of media time present in `after` but not in `before`.
  function newSpan(before, after) {
    let s = Infinity, e = -Infinity;
    for (const [a0, a1] of after) {
      let cur = a0;
      const cuts = before.filter(([b0, b1]) => b1 > a0 && b0 < a1).sort((x, y) => x[0] - y[0]);
      for (const [b0, b1] of cuts) {
        if (b0 - cur > 0.02) { s = Math.min(s, cur); e = Math.max(e, b0); }
        cur = Math.max(cur, b1);
      }
      if (a1 - cur > 0.02) { s = Math.min(s, cur); e = Math.max(e, a1); }
    }
    return s < e ? [s, e] : null;
  }

  // ---------------------------------------------------------------------------
  // Sessions: one per MediaSource (a video, or an ad)
  // ---------------------------------------------------------------------------

  let nextId = 1;
  const sessions = new Map();        // blob URL -> Session
  const sessionByMs = new WeakMap(); // MediaSource -> Session
  const trackers = new WeakMap();    // SourceBuffer -> Tracker

  class Session {
    constructor(ms) {
      this.id = nextId++;
      this.ms = ms;
      this.url = null;
      this.videoId = new URLSearchParams(location.search).get("v") || location.pathname;
      this.chunks = [];   // {id, start, end, state: queued|sending|done|error, bytes, mime, buffer}
      this.ended = false;
    }
    covered(start, end) {
      let c = 0;
      for (const ch of this.chunks) {
        if (ch.state === "error") continue;
        c += Math.max(0, Math.min(end, ch.end) - Math.max(start, ch.start));
      }
      return c >= (end - start) * 0.9;
    }
    chunkAt(t) {
      return this.chunks.find((c) => c.start - 0.05 <= t && t < c.end - 0.02);
    }
  }

  class Tracker {
    constructor(sb, mime, session) {
      this.sb = sb;
      this.mime = mime;
      this.isMp4 = /mp4/i.test(mime);
      this.session = session;
      this.init = null;
      this.pending = null; // {init, parts: [{bytes, start, end, clean}], start, end}
      this.lastAppend = 0;
    }

    onAppend(data) {
      if (!settings.enabled) return;
      const bytes = toBytes(data);
      const before = rangesOf(this.sb.buffered);
      const done = () => {
        this.sb.removeEventListener("updateend", done);
        this.onAppended(bytes, before, rangesOf(this.sb.buffered));
      };
      this.sb.addEventListener("updateend", done);
    }

    onAppended(bytes, before, after) {
      this.lastAppend = performance.now();
      let media = bytes, clean;
      const split = splitInit(bytes, this.isMp4);
      if (split > 0) {
        this.init = bytes.slice(0, split);
        media = bytes.subarray(split);
        clean = true;
      } else {
        clean = split === 0;
      }
      if (!this.init || media.length === 0) return;
      const span = newSpan(before, after);
      if (!span) return; // re-append of data already buffered

      const p = this.pending;
      const contiguous = p && p.init === this.init && Math.abs(span[0] - p.end) < 0.1;
      if (!contiguous) {
        if (p) this.flush(p.parts.length);
        this.pending = null;
        if (!clean) return; // can't decode a chunk that starts mid-cluster
        this.pending = { init: this.init, parts: [], start: span[0], end: span[0] };
      }
      const q = this.pending;
      q.parts.push({ bytes: media, start: span[0], end: span[1], clean });
      q.end = span[1];

      // Cut at the last clean segment boundary once we have a full chunk.
      const target = this.session.covered(q.start - 0.5, q.start)
        ? settings.chunkSeconds : Math.min(settings.firstChunkSeconds, settings.chunkSeconds);
      if (q.end - q.start >= target) {
        let k = q.parts.length - 1;
        while (k > 0 && !q.parts[k].clean) k--;
        if (k > 0) this.flush(k);
        else if (q.end - q.start > target * 2) this.flush(q.parts.length);
      }
    }

    // Emit parts[0..k) as a chunk; the rest stay pending.
    flush(k) {
      const q = this.pending;
      if (!q || k === 0) return;
      const parts = q.parts.slice(0, k);
      q.parts = q.parts.slice(k);
      q.start = q.parts.length ? q.parts[0].start : q.end;
      if (!q.parts.length) this.pending = null;

      const start = parts[0].start, end = parts[parts.length - 1].end;
      if (this.session.covered(start, end)) return;
      let len = q.init.length;
      for (const p of parts) len += p.bytes.length;
      const buf = new Uint8Array(len);
      buf.set(q.init, 0);
      let off = q.init.length;
      for (const p of parts) { buf.set(p.bytes, off); off += p.bytes.length; }
      this.session.chunks.push({
        id: `${this.session.id}-${start.toFixed(2)}`,
        start, end, state: "queued", bytes: buf, mime: this.mime, buffer: null, retryAt: 0,
      });
      log(`chunk ${start.toFixed(1)}-${end.toFixed(1)}s (${(len / 1024) | 0} KiB) queued`);
      pump();
    }

    // Flush a short pending chunk if nothing more is arriving and playback needs it.
    maybeIdleFlush(video) {
      const q = this.pending;
      if (!q || !q.parts.length) return;
      const idle = performance.now() - this.lastAppend > 1500;
      const needed = this.session.ended || (video && q.start < video.currentTime + 10);
      if (idle && needed) this.flush(q.parts.length);
    }
  }

  // ---------------------------------------------------------------------------
  // Hooks
  // ---------------------------------------------------------------------------

  const MS = window.MediaSource || window.ManagedMediaSource;
  if (!MS) return;

  const origCreateURL = URL.createObjectURL;
  URL.createObjectURL = function (obj) {
    const url = origCreateURL.apply(this, arguments);
    // The URL is usually created before any SourceBuffer exists, so register now.
    const isMs = (window.MediaSource && obj instanceof window.MediaSource) ||
                 (window.ManagedMediaSource && obj instanceof window.ManagedMediaSource);
    if (isMs) { const s = sessionFor(obj); s.url = url; sessions.set(url, s); }
    return url;
  };

  function sessionFor(ms) {
    let s = sessionByMs.get(ms);
    if (!s) { s = new Session(ms); sessionByMs.set(ms, s); }
    return s;
  }

  for (const Ctor of [window.MediaSource, window.ManagedMediaSource]) {
    if (!Ctor) continue;
    const proto = Ctor.prototype;
    const origAdd = proto.addSourceBuffer;
    proto.addSourceBuffer = function (mime) {
      const sb = origAdd.apply(this, arguments);
      if (/^audio\//i.test(mime)) trackers.set(sb, new Tracker(sb, mime, sessionFor(this)));
      return sb;
    };
    const origEos = proto.endOfStream;
    proto.endOfStream = function () {
      const s = sessionByMs.get(this);
      if (s) s.ended = true;
      return origEos.apply(this, arguments);
    };
  }

  for (const Ctor of [window.SourceBuffer, window.ManagedSourceBuffer]) {
    if (!Ctor) continue;
    const proto = Ctor.prototype;
    const origAppend = proto.appendBuffer;
    proto.appendBuffer = function (data) {
      const t = trackers.get(this);
      if (t) { try { t.onAppend(data); } catch (e) { log("capture error", e); } }
      return origAppend.apply(this, arguments);
    };
    const origChangeType = proto.changeType;
    if (origChangeType) {
      proto.changeType = function (mime) {
        const t = trackers.get(this);
        if (t) { t.mime = mime; t.isMp4 = /mp4/i.test(mime); t.init = null; }
        return origChangeType.apply(this, arguments);
      };
    }
  }

  // ---------------------------------------------------------------------------
  // Talking to the server (through bridge.js -> background.js)
  // ---------------------------------------------------------------------------

  let inFlight = null;
  let serverError = null;
  const allSessions = () => [...sessions.values()];

  function currentSession() {
    const v = mainVideo();
    return v ? sessions.get(v.currentSrc || v.src) : null;
  }

  // Send one chunk at a time, the one playback needs soonest first.
  function pump() {
    if (inFlight || !settings.enabled) return;
    const s = currentSession();
    if (!s) return;
    const t = mainVideo().currentTime;
    const now = performance.now();
    const queued = s.chunks.filter((c) => c.state === "queued" && c.retryAt <= now);
    if (!queued.length) return;
    const score = (c) => (c.end <= t ? 1e6 + c.start : Math.max(0, c.start - t));
    queued.sort((a, b) => score(a) - score(b));
    const c = queued[0];
    c.state = "sending";
    inFlight = c;
    c.sentAt = now;
    // Bytes are copied (not transferred) so the chunk can be retried on failure.
    window.postMessage({ [TAG]: "separate", id: c.id, mime: c.mime, bytes: c.bytes }, "*");
  }

  window.addEventListener("message", async (ev) => {
    if (ev.source !== window || !ev.data || !ev.data[TAG]) return;
    const m = ev.data;
    if (m[TAG] === "settings") {
      const wasEnabled = settings.enabled;
      Object.assign(settings, m.settings);
      if (wasEnabled && !settings.enabled) stopProcessed();
      return;
    }
    if (m[TAG] !== "result") return;
    let chunk = null;
    for (const s of allSessions()) chunk = chunk || s.chunks.find((c) => c.id === m.id);
    if (inFlight && inFlight.id === m.id) inFlight = null;
    if (!chunk) return pump();
    if (m.error) {
      // Server down / failed: retry later. In "wait" mode the video stays paused.
      chunk.state = "queued";
      chunk.retryAt = performance.now() + 3000;
      serverError = m.error;
      log("server error", m.error);
    } else {
      try {
        chunk.buffer = await ensureCtx().decodeAudioData(m.audio);
        chunk.state = "done";
        chunk.bytes = null;
        serverError = null;
        log(`chunk ${chunk.start.toFixed(1)}s ready in ${((performance.now() - chunk.sentAt) / 1000).toFixed(1)}s`);
      } catch (e) {
        chunk.state = "error";
        serverError = "could not decode processed audio";
      }
    }
    pump();
    tick();
  });

  // ---------------------------------------------------------------------------
  // Playback
  // ---------------------------------------------------------------------------

  let ctx = null;
  const routed = new WeakMap(); // video -> GainNode for the original audio
  let current = null;           // {chunk, node, ctxStart, mediaStart, rate}
  let next = null;              // pre-scheduled following chunk
  let autoPaused = false;

  function ensureCtx() {
    if (!ctx) ctx = new AudioContext({ latencyHint: "playback" });
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    return ctx;
  }
  // AudioContext may only start after a user gesture.
  for (const e of ["pointerdown", "keydown"]) {
    window.addEventListener(e, () => { if (ctx) ctx.resume().catch(() => {}); }, true);
  }

  function mainVideo() {
    return document.querySelector("video.html5-main-video") || document.querySelector("video");
  }

  function originalGain(video) {
    let g = routed.get(video);
    if (!g) {
      const c = ensureCtx();
      if (c.state !== "running") return null; // routing now would silence the video
      const src = c.createMediaElementSource(video);
      g = c.createGain();
      src.connect(g).connect(c.destination);
      routed.set(video, g);
    }
    return g;
  }

  function startNode(chunk, when, offset, rate) {
    const node = ctx.createBufferSource();
    node.buffer = chunk.buffer;
    node.playbackRate.value = rate;
    node.connect(ctx.destination);
    node.start(when, Math.max(0, offset));
    return node;
  }

  function stopNode(p) { if (p) { try { p.node.stop(); } catch (_) {} p.node.disconnect(); } }
  function stopProcessed() { stopNode(current); stopNode(next); current = next = null; }

  function mediaTimeNow(p) {
    return p.mediaStart + (ctx.currentTime - p.ctxStart) * p.rate;
  }

  function tick() {
    const video = mainVideo();
    const session = currentSession();
    if (session) for (const tr of trackersOf(session)) tr.maybeIdleFlush(video);
    pump();
    updateBadge(video, session);

    if (!video || !settings.enabled || !session) {
      stopProcessed();
      const g = video && routed.get(video);
      if (g) g.gain.value = 1;
      return;
    }
    const g = originalGain(video);
    if (!g) return;
    const t = video.currentTime;
    const rate = video.playbackRate;
    const chunk = session.chunkAt(t);
    const ready = chunk && chunk.state === "done";

    if (ready) {
      g.gain.value = 0;
      if (autoPaused) { autoPaused = false; video.play().catch(() => {}); }
    } else {
      g.gain.value = settings.mode === "original" ? 1 : 0;
      if (settings.mode === "wait" && !video.paused && !video.ended) {
        autoPaused = true;
        video.pause();
      }
    }

    if (!ready || video.paused || video.seeking || video.ended) { stopProcessed(); return; }

    // Promote the pre-scheduled next chunk once playback has reached it.
    if (current && current.chunk !== chunk && next && next.chunk === chunk) {
      stopNode(current);
      current = next;
      next = null;
    }
    if (current && (current.chunk !== chunk || current.rate !== rate ||
        Math.abs(mediaTimeNow(current) - t) > 0.08)) {
      stopProcessed();
    }
    if (!current) {
      ensureCtx();
      const when = ctx.currentTime + 0.02;
      const mediaStart = t + 0.02 * rate;
      current = { chunk, ctxStart: when, mediaStart, rate,
                  node: startNode(chunk, when, mediaStart - chunk.start, rate) };
    }
    // Schedule the following chunk sample-accurately so boundaries are seamless.
    const following = session.chunks.find((c) => c.state === "done" && Math.abs(c.start - chunk.end) < 0.1);
    if (following && (!next || next.chunk !== following)) {
      stopNode(next);
      const when = current.ctxStart + (following.start - current.mediaStart) / rate;
      if (when > ctx.currentTime + 0.01) {
        next = { chunk: following, ctxStart: when, mediaStart: following.start, rate,
                 node: startNode(following, when, 0, rate) };
      }
    }
  }

  function trackersOf(session) {
    // SourceBuffers aren't enumerable from a WeakMap; read them from the MediaSource.
    const out = [];
    try {
      for (const sb of session.ms.sourceBuffers) { const t = trackers.get(sb); if (t) out.push(t); }
    } catch (_) {}
    return out;
  }

  // ---------------------------------------------------------------------------
  // Status badge
  // ---------------------------------------------------------------------------

  let badge = null;
  function updateBadge(video, session) {
    const player = document.querySelector("#movie_player") || (video && video.parentElement);
    if (!player) return;
    if (!badge || !badge.isConnected) {
      badge = document.createElement("div");
      badge.style.cssText = "position:absolute;top:12px;left:12px;z-index:60;padding:4px 10px;" +
        "border-radius:12px;font:500 12px Roboto,Arial,sans-serif;color:#fff;" +
        "background:rgba(0,0,0,.65);pointer-events:none;transition:opacity .3s";
      player.appendChild(badge);
    }
    let text = "", show = true;
    if (!settings.enabled) show = false;
    else if (!session) text = "Music remover: waiting for audio…";
    else if (serverError) text = "Music remover: server error – " + serverError.slice(0, 80);
    else {
      const c = session.chunkAt(video.currentTime);
      const done = session.chunks.filter((x) => x.state === "done").length;
      if (c && c.state === "done") {
        text = `Music removed ✓ (${done} chunk${done === 1 ? "" : "s"} ready)`;
        show = video.paused; // stay out of the way while watching
      } else if (autoPaused) text = "Removing music… paused until ready";
      else text = "Removing music…";
    }
    badge.textContent = text;
    badge.style.opacity = show && text ? "1" : "0";
  }

  setInterval(tick, 100);
  document.addEventListener("seeked", tick, true);
  document.addEventListener("play", tick, true);
  document.addEventListener("pause", stopProcessed, true);
  document.addEventListener("ratechange", tick, true);

  window.postMessage({ [TAG]: "hello" }, "*");
})();
