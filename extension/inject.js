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

  // When the extension is installed, reloaded or updated while a YouTube tab is open,
  // background.js injects this script again, and the previous copy is still running in
  // the page. Stop it and take over its audio routing: a <video> can only be connected
  // to WebAudio once.
  const prev = window.__musicremover;
  if (prev && typeof prev.dispose === "function") { try { prev.dispose(); } catch (_) {} }
  let disposed = false;

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
    // Chime when a video that was held for processing can start playing.
    readySound: true,
  };

  // console.log (not debug) so the lines show at Chrome's default console level.
  const log = (...a) => console.log("[musicremover]", ...a);

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

    // Returns a cancel function for when the append is rejected (e.g. QuotaExceededError
    // when the buffer is full; YouTube then evicts and retries). Without cancelling,
    // the listener would fire for the *next* append and record the wrong bytes.
    onAppend(data) {
      if (!settings.enabled) return null;
      const bytes = toBytes(data);
      const before = rangesOf(this.sb.buffered);
      let aborted = false;
      const onAbort = () => { aborted = true; };
      const cleanup = () => {
        this.sb.removeEventListener("updateend", done);
        this.sb.removeEventListener("abort", onAbort);
      };
      const done = () => {
        cleanup();
        // An aborted append may have buffered only part of the data: skip it.
        if (!aborted) this.onAppended(bytes, before, rangesOf(this.sb.buffered));
      };
      this.sb.addEventListener("updateend", done);
      this.sb.addEventListener("abort", onAbort);
      return cleanup;
    }

    onAppended(bytes, before, after) {
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
      // Only new audio counts as activity, so re-appends can't hold off an idle flush.
      this.lastAppend = performance.now();

      const p = this.pending;
      // Tolerate small overlaps/gaps between pieces (YouTube's streaming appends in
      // small bursts whose ranges don't always line up exactly).
      const contiguous = p && p.init === this.init && span[0] >= p.end - 0.5 && span[0] - p.end < 0.25;
      if (!contiguous) {
        if (p) this.flush(p.parts.length, "discontinuity (seek or format change)");
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
        // Last clean boundary that leaves at least 1 s of new (not carried) audio before it.
        let k = q.parts.length - 1;
        while (k > 0 && !(q.parts[k].clean && !q.parts[k].carry && q.parts[k - 1].end > q.start + 1)) k--;
        if (k > 0) this.flush(k, `reached ${target}s`);
        else if (q.end - q.start > target * 2) this.flush(q.parts.length, "no clean cut point");
      }
    }

    // Emit parts[0..k) as a chunk covering [q.start, parts[k-1].end); the rest stay pending.
    //
    // A chunk must start at a cluster/fragment boundary to be decodable, but YouTube
    // often continues right after a flush with the *middle* of a cluster. Dropping that
    // would leave a hole no chunk ever covers, and "pause until ready" would wait
    // forever. So when everything is flushed, the bytes from the last clean boundary are
    // carried into the next chunk: its audio then starts a little earlier (audioStart)
    // than the time range it covers (start), and playback skips that lead-in.
    flush(k, reason = "") {
      const q = this.pending;
      if (!q || k === 0) return;
      const parts = q.parts.slice(0, k);
      const rest = q.parts.slice(k);
      const start = q.start, end = parts[parts.length - 1].end;
      const audioStart = parts[0].start;
      if (rest.length) {
        q.parts = rest; // k is always a clean boundary here
        q.start = rest[0].start;
      } else {
        let c = parts.length - 1;
        while (c > 0 && !parts[c].clean) c--;
        const carry = parts.slice(c);
        if (end - carry[0].start <= 15) {
          q.parts = carry.map((p) => ({ ...p, carry: true }));
          q.start = end; // the next chunk covers from where this one ends
        } else {
          this.pending = null; // no boundary for too long; a small gap is the lesser evil
        }
      }

      if (end - start < 0.05 || this.session.covered(start, end)) return;
      let len = q.init.length;
      for (const p of parts) len += p.bytes.length;
      const buf = new Uint8Array(len);
      buf.set(q.init, 0);
      let off = q.init.length;
      for (const p of parts) { buf.set(p.bytes, off); off += p.bytes.length; }
      this.session.chunks.push({
        id: `${this.session.id}-${start.toFixed(2)}`,
        start, end, audioStart, state: "queued", bytes: buf, mime: this.mime, buffer: null, retryAt: 0,
      });
      log(`chunk ${start.toFixed(1)}-${end.toFixed(1)}s (${(len / 1024) | 0} KiB) queued: ${reason}`);
      pump();
    }

    // Send a short chunk early only when playback is actually blocked on it and
    // YouTube has stopped delivering. YouTube streams in bursts with pauses of a
    // few seconds between them; flushing on every pause produced ~1 s chunks, and
    // each request costs the same fixed model time, so tiny chunks fall behind.
    maybeIdleFlush(video) {
      const q = this.pending;
      if (!q || !q.parts.length) return;
      const dur = q.end - q.start; // only new audio; a carried lead-in doesn't count
      if (dur <= 0) return;
      if (this.session.ended) return this.flush(q.parts.length, "end of video");
      if (!video) return;
      const idle = performance.now() - this.lastAppend;
      const ahead = q.start - video.currentTime; // how soon playback reaches this audio
      // Send early enough that processing finishes before playback gets there.
      const lead = Math.min(30, Math.max(4, 1.5 * procSeconds + 2));
      if (ahead <= lead && idle > 1000 && dur >= 3) {
        this.flush(q.parts.length, `playback reaches it in ${Math.max(0, ahead).toFixed(1)}s`);
      } else if (ahead <= 0.5 && idle > 4000 && dur > 0.5) {
        this.flush(q.parts.length, "playback waiting on it");
      } else if (ahead <= 2 && idle > 10000 && dur > 0.5) {
        this.flush(q.parts.length, "YouTube stopped buffering");
      }
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
      let cancel = null;
      if (t) { try { cancel = t.onAppend(data); } catch (e) { log("capture error", e); } }
      try {
        return origAppend.apply(this, arguments);
      } catch (e) {
        if (cancel) cancel();
        throw e; // YouTube handles this itself (evicts old data and retries)
      }
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
  let procSeconds = 3; // running estimate of how long the server takes per chunk
  const allSessions = () => [...sessions.values()];

  function currentSession() {
    const v = mainVideo();
    return v ? sessions.get(v.currentSrc || v.src) : null;
  }

  // Send one chunk at a time, the one playback needs soonest first.
  function pump() {
    // Watchdog: a request that never answered (e.g. the message was lost) is retried.
    if (inFlight && performance.now() - inFlight.sentAt > 20 * 60 * 1000) {
      log(`no answer for chunk ${inFlight.start.toFixed(1)}s after 20 min; retrying`);
      inFlight.state = "queued";
      inFlight = null;
    }
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
    if (disposed || ev.source !== window || !ev.data || !ev.data[TAG]) return;
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
        procSeconds = 0.7 * procSeconds + 0.3 * (performance.now() - chunk.sentAt) / 1000;
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

  let ctx = (prev && prev.ctx) || null;
  const routed = (prev && prev.routed) || new WeakMap(); // video -> GainNode for the original audio
  let current = null;           // {chunk, node, ctxStart, mediaStart, rate}
  let next = null;              // pre-scheduled following chunk
  let autoPaused = false;
  let resumeTimer = 0;
  let autoPausedAt = 0;
  let mutedUntil = -1; // media time until which playback continues muted (stall safety net)
  let needsReload = false; // video was loaded before the script and can't be restarted

  // Two-note "ready" chime, synthesized so no audio file is needed.
  // Returns how long to wait (ms) before resuming playback.
  function playReadySound() {
    const c = ensureCtx();
    if (c.state !== "running") return 0;
    const t0 = c.currentTime + 0.02;
    [[659.25, 0], [987.77, 0.12]].forEach(([freq, at]) => {
      const osc = c.createOscillator();
      const env = c.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      env.gain.setValueAtTime(0, t0 + at);
      env.gain.linearRampToValueAtTime(0.18, t0 + at + 0.015);
      env.gain.exponentialRampToValueAtTime(0.0001, t0 + at + 0.45);
      osc.connect(env).connect(c.destination);
      osc.start(t0 + at);
      osc.stop(t0 + at + 0.5);
    });
    return 550;
  }

  function ensureCtx() {
    if (!ctx) ctx = new AudioContext({ latencyHint: "playback" });
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    return ctx;
  }
  // AudioContext may only start after a user gesture.
  for (const e of ["pointerdown", "keydown"]) {
    window.addEventListener(e, () => { if (ctx) ctx.resume().catch(() => {}); }, true);
  }

  function isVisible(v) {
    const r = v.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight;
  }

  // The <video> being watched. Regular videos have one; Shorts can have several
  // (the one on screen plus preloaded ones), so prefer a captured video that is
  // playing, then one that is on screen.
  function mainVideo() {
    const vids = [...document.querySelectorAll("video")];
    const known = vids.filter((v) => sessions.has(v.currentSrc || v.src));
    return known.find((v) => !v.paused && isVisible(v)) || known.find(isVisible) || known[0] ||
      document.querySelector("#shorts-player video, video.html5-main-video") || vids[0] || null;
  }

  function playerOf(video) {
    return (video && video.closest("#movie_player, #shorts-player, .html5-video-player")) ||
      document.querySelector("#movie_player") || (video && video.parentElement);
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

  // Is audio for time t on its way (captured but not sent, or being processed)?
  function coming(session, t) {
    if (session.chunks.some((c) => (c.state === "queued" || c.state === "sending") &&
        c.start - 0.5 <= t && t < c.end)) return true;
    return trackersOf(session).some((tr) => tr.pending && tr.pending.start <= t + 0.5 && tr.pending.end > t);
  }

  function tick() {
    if (disposed) return;
    const video = mainVideo();
    const session = currentSession();
    if (session) for (const tr of trackersOf(session)) tr.maybeIdleFlush(video);
    pump();
    updateBadge(video, session);
    updateProgress(video, session);

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
      if (autoPaused && !resumeTimer) {
        // Chime first, then start playback once it has rung out. Brief holds
        // (under 2 s) resume silently so the chime doesn't become noise.
        const heldLong = performance.now() - autoPausedAt > 2000;
        const delay = settings.readySound && heldLong ? playReadySound() : 0;
        resumeTimer = setTimeout(() => {
          resumeTimer = 0;
          autoPaused = false;
          video.play().catch(() => {});
        }, delay);
      }
    } else {
      g.gain.value = settings.mode === "original" ? 1 : 0;
      if (settings.mode === "wait" && !video.paused && !video.ended && !(t < mutedUntil)) {
        autoPaused = true;
        autoPausedAt = performance.now();
        video.pause();
      }
      // Safety net: if nothing for this spot has been captured or is being processed
      // after 20 s, it never will be. Play this stretch muted instead of hanging forever.
      if (autoPaused && !resumeTimer && performance.now() - autoPausedAt > 20000 && !coming(session, t)) {
        const nextChunk = session.chunks.filter((c) => c.start > t).sort((a, b) => a.start - b.start)[0];
        mutedUntil = nextChunk ? nextChunk.start : t + 10;
        log(`no audio captured for ${t.toFixed(1)}s; playing muted until ${mutedUntil.toFixed(1)}s`);
        autoPaused = false;
        video.play().catch(() => {});
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
                  node: startNode(chunk, when, mediaStart - chunk.audioStart, rate) };
    }
    // Schedule the following chunk sample-accurately so boundaries are seamless.
    const following = session.chunks.find((c) => c.state === "done" && Math.abs(c.start - chunk.end) < 0.1);
    if (following && (!next || next.chunk !== following)) {
      stopNode(next);
      const when = current.ctxStart + (following.start - current.mediaStart) / rate;
      if (when > ctx.currentTime + 0.01) {
        next = { chunk: following, ctxStart: when, mediaStart: following.start, rate,
                 node: startNode(following, when, following.start - following.audioStart, rate) };
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
    const player = playerOf(video);
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
    else if (needsReload && !session) text = "Music remover: reload the page to remove music from this video";
    else if (!session) text = "Music remover: waiting for audio…";
    else if (serverError) text = "Music remover: server error – " + serverError.slice(0, 80);
    else {
      const c = session.chunkAt(video.currentTime);
      const done = session.chunks.filter((x) => x.state === "done").length;
      if (c && c.state === "done") {
        text = `Music removed ✓ (${done} chunk${done === 1 ? "" : "s"} ready)`;
        show = video.paused; // stay out of the way while watching
      } else if (video.currentTime < mutedUntil) text = "Music remover: this part couldn't be captured – playing muted";
      else if (autoPaused) text = "Removing music… paused until ready";
      else text = "Removing music…";
    }
    badge.textContent = text;
    badge.style.opacity = show && text ? "1" : "0";
  }

  // ---------------------------------------------------------------------------
  // Progress on the play bar: which parts are processed, and how far ahead
  // ---------------------------------------------------------------------------

  const COLORS = { done: "#2ecc71", sending: "#f5a623", queued: "rgba(255,255,255,.55)" };
  let bar = null, label = null, barKey = "";

  function fmtTime(s) {
    s = Math.max(0, Math.floor(s));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = String(s % 60).padStart(2, "0");
    return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
  }

  // End of the processed stretch that playback is in (or about to enter).
  function removedUpTo(session, t) {
    let reach = t;
    for (;;) {
      const c = session.chunks.find((x) => x.state === "done" && x.start - 0.1 <= reach && x.end > reach + 0.01);
      if (!c) return reach;
      reach = c.end;
    }
  }

  // Video length: from the element, or from YouTube's player while the stream is open.
  function videoDuration(video, player) {
    if (video && video.duration > 0 && isFinite(video.duration)) return video.duration;
    const d = player && typeof player.getDuration === "function" ? player.getDuration() : 0;
    return d > 0 && isFinite(d) ? d : 0;
  }

  function updateProgress(video, session) {
    const player = playerOf(video);
    const dur = videoDuration(video, player);
    if (!player || !session || !settings.enabled || !dur) {
      if (bar) bar.style.display = "none";
      if (label) label.style.display = "none";
      return;
    }
    // Regular videos: sit just above YouTube's progress bar (it hides with the controls).
    // Shorts and anything else: our own strip at the bottom of the video.
    const ytBar = player.querySelector(".ytp-progress-bar");
    const host = ytBar || player;
    if (!bar || !bar.isConnected || bar.parentElement !== host) {
      if (bar) bar.remove();
      bar = document.createElement("div");
      bar.title = "Music remover: green = music removed, amber = processing, grey = waiting";
      bar.style.cssText = "position:absolute;left:0;right:0;height:4px;z-index:45;pointer-events:none;" +
        (ytBar ? "top:-7px;" : "bottom:0;background:rgba(0,0,0,.35);");
      host.appendChild(bar);
      barKey = "";
    }
    bar.style.display = "";

    const key = session.chunks.map((c) => `${c.start.toFixed(1)}-${c.end.toFixed(1)}:${c.state}`).join(",") + "|" + dur;
    if (key !== barKey) {
      barKey = key;
      bar.textContent = "";
      for (const c of session.chunks) {
        const color = COLORS[c.state];
        if (!color) continue;
        const seg = document.createElement("div");
        seg.style.cssText = `position:absolute;top:0;bottom:0;background:${color};` +
          `left:${(100 * c.start / dur).toFixed(3)}%;width:${(100 * (c.end - c.start) / dur).toFixed(3)}%`;
        bar.appendChild(seg);
      }
    }

    // "Music removed to 2:35" next to YouTube's time display (or on our strip).
    const t = video.currentTime;
    const reach = removedUpTo(session, Math.max(0, t - 0.2));
    const text = reach >= dur - 0.5 ? "· music removed to the end" :
      reach > t + 0.5 ? `· music removed to ${fmtTime(reach)}` :
      session.chunks.some((c) => c.state === "sending") ? "· removing music…" : "· music remover waiting";
    const timeDisplay = player.querySelector(".ytp-time-display");
    const labelHost = timeDisplay || bar;
    if (!label || !label.isConnected || label.parentElement !== labelHost) {
      if (label) label.remove();
      label = document.createElement("span");
      label.style.cssText = timeDisplay ? "margin-left:6px;color:#2ecc71;" :
        "position:absolute;right:6px;bottom:6px;font:500 11px Roboto,Arial,sans-serif;color:#fff;" +
        "background:rgba(0,0,0,.6);padding:2px 6px;border-radius:8px;white-space:nowrap";
      labelHost.appendChild(label);
    }
    label.style.display = "";
    if (label.textContent !== text) label.textContent = text;
  }

  // ---------------------------------------------------------------------------
  // Start-up
  // ---------------------------------------------------------------------------

  const timer = setInterval(tick, 100);
  document.addEventListener("seeked", tick, true);
  document.addEventListener("play", tick, true);
  document.addEventListener("pause", () => { if (!disposed) stopProcessed(); }, true);
  document.addEventListener("ratechange", tick, true);

  window.__musicremover = {
    get ctx() { return ctx; },
    routed,
    dispose() {
      disposed = true;
      clearInterval(timer);
      clearTimeout(resumeTimer);
      settings.enabled = false; // the old appendBuffer wrappers stay installed but go idle
      stopProcessed();
      for (const el of [badge, bar, label]) if (el) el.remove();
    },
  };

  // A video that was already playing before this script ran (the tab was open when the
  // extension was installed or reloaded) created its MediaSource before our hooks, so its
  // audio can't be captured. Restart it at the same position through YouTube's player API;
  // the reloaded stream goes through the hooks.
  function captureAlreadyLoaded() {
    const v = document.querySelector("#shorts-player video, video.html5-main-video") || document.querySelector("video");
    const src = v && (v.currentSrc || v.src);
    if (!src || !src.startsWith("blob:") || sessions.has(src)) return;
    const player = playerOf(v);
    const data = player && typeof player.getVideoData === "function" && player.getVideoData();
    if (!data || !data.video_id || typeof player.loadVideoById !== "function") {
      needsReload = true;
      return;
    }
    const t = typeof player.getCurrentTime === "function" ? player.getCurrentTime() : v.currentTime;
    log(`restarting the already-loaded video at ${t.toFixed(1)}s so its audio can be captured`);
    player.loadVideoById({ videoId: data.video_id, startSeconds: t });
  }
  setTimeout(captureAlreadyLoaded, 300);

  window.postMessage({ [TAG]: "hello" }, "*");
})();
