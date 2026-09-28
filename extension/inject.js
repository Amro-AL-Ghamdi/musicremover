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
  // A copy from before hand-over existed can't be stopped, and two copies fight over the
  // video. Leave it running until the page is reloaded.
  if (!prev && window.SourceBuffer && /trackers\.get\(this\)/.test(String(SourceBuffer.prototype.appendBuffer))) {
    console.log("[musicremover] an older version is still active in this tab; reload the page to use the new one");
    return;
  }
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
  // Container parsing: split the appended byte stream into self-contained units
  // (WebM clusters / MP4 moof+mdat fragments), each with its own timestamp.
  //
  // Timing comes from the media itself, never from SourceBuffer.buffered: on real
  // YouTube the growth of `buffered` doesn't line up with individual appends, and
  // guessing from it left most of every segment uncaptured.
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

  function readUint(b, pos, len) {
    let v = 0;
    for (let i = 0; i < len; i++) v = v * 256 + b[pos + i];
    return v;
  }

  const EBML_ID = 0x1a45dfa3, SEGMENT_ID = 0x18538067, CLUSTER_ID = 0x1f43b675;
  const TIMECODE_ID = 0xe7, SIMPLEBLOCK_ID = 0xa3, BLOCKGROUP_ID = 0xa0, BLOCK_ID = 0xa1;
  // Level-1 elements: seeing one of these ends an unknown-size cluster.
  const TOP_LEVEL = new Set([EBML_ID, SEGMENT_ID, CLUSTER_ID, 0x1c53bb6b /* Cues */,
    0x1254c367 /* Tags */, 0x1043a770 /* Chapters */, 0x1941a469 /* Attachments */,
    0x114d9b74 /* SeekHead */, 0x1549a966 /* Info */, 0x1654ae6b /* Tracks */]);

  // WebM init segment (EBML header + Segment header + children up to the first
  // Cluster). Returns its length, or -1 if more bytes are needed.
  function webmInitLength(b, pos, final) {
    const id = readVint(b, pos, true), hsz = id && readVint(b, pos + id.len, false);
    if (!hsz) return -1;
    let p = pos + id.len + hsz.len + hsz.val;
    const seg = readVint(b, p, true), ssz = seg && readVint(b, p + seg.len, false);
    if (!ssz) return -1;
    p += seg.len + ssz.len;
    for (;;) {
      if (p >= b.length) return final ? p - pos : -1;
      const cid = readVint(b, p, true), csz = cid && readVint(b, p + cid.len, false);
      if (!csz) return final ? p - pos : -1;
      if (cid.val === CLUSTER_ID) return p - pos;
      p += cid.len + csz.len + csz.val;
    }
  }

  function webmTimecodeScale(init) {
    for (let i = 0; i + 3 < init.length; i++) {
      if (init[i] === 0x2a && init[i + 1] === 0xd7 && init[i + 2] === 0xb1) {
        const sz = readVint(init, i + 3, false);
        if (sz && sz.val <= 8) return readUint(init, i + 3 + sz.len, sz.val);
      }
    }
    return 1e6; // WebM default: 1 ms
  }

  // End position of a cluster starting at pos, or -1 if it isn't complete yet.
  function webmClusterEnd(b, pos, final) {
    const id = readVint(b, pos, true), sz = id && readVint(b, pos + id.len, false);
    if (!sz) return -1;
    const body = pos + id.len + sz.len;
    if (!sz.unknown) return body + sz.val <= b.length ? body + sz.val : -1;
    // Unknown size (streaming): the cluster ends where the next level-1 element starts.
    let p = body;
    for (;;) {
      if (p >= b.length) return final ? b.length : -1;
      const cid = readVint(b, p, true);
      if (!cid) return final ? b.length : -1;
      if (TOP_LEVEL.has(cid.val)) return p;
      const csz = readVint(b, p + cid.len, false);
      if (!csz || csz.unknown) return final ? b.length : -1;
      p += cid.len + csz.len + csz.val;
      if (p > b.length) return final ? b.length : -1;
    }
  }

  // Cluster timecode and the largest block offset in it (both in timecode units).
  function webmClusterTimes(b, pos, end) {
    const id = readVint(b, pos, true), sz = readVint(b, pos + id.len, false);
    let p = pos + id.len + sz.len, tc = null, maxRel = 0;
    const blockRel = (q) => {
      const tn = readVint(b, q, false);
      if (!tn) return 0;
      let v = (b[q + tn.len] << 8) | b[q + tn.len + 1];
      return v & 0x8000 ? v - 0x10000 : v;
    };
    while (p < end) {
      const cid = readVint(b, p, true), csz = cid && readVint(b, p + cid.len, false);
      if (!csz) break;
      const data = p + cid.len + csz.len;
      if (cid.val === TIMECODE_ID) tc = readUint(b, data, csz.val);
      else if (cid.val === SIMPLEBLOCK_ID) maxRel = Math.max(maxRel, blockRel(data));
      else if (cid.val === BLOCKGROUP_ID) {
        for (let q = data; q < data + csz.val;) {
          const gid = readVint(b, q, true), gsz = gid && readVint(b, q + gid.len, false);
          if (!gsz) break;
          if (gid.val === BLOCK_ID) maxRel = Math.max(maxRel, blockRel(q + gid.len + gsz.len));
          q += gid.len + gsz.len + gsz.val;
        }
      }
      if (csz.unknown) break;
      p = data + csz.val;
    }
    return { tc, maxRel };
  }

  // MP4 box header at pos: {type, size (whole box), hdr} or null if incomplete.
  function mp4Box(b, pos) {
    if (pos + 8 > b.length) return null;
    let size = readUint(b, pos, 4), hdr = 8;
    const type = String.fromCharCode(b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]);
    if (size === 1) {
      if (pos + 16 > b.length) return null;
      size = readUint(b, pos + 8, 8);
      hdr = 16;
    } else if (size === 0) size = b.length - pos;
    return { type, size, hdr };
  }

  // Find a box by path (e.g. ["traf", "tfdt"]) inside [pos, end). Returns its content offset.
  function mp4Find(b, pos, end, path) {
    for (let p = pos; p + 8 <= end;) {
      const box = mp4Box(b, p);
      if (!box || box.size < 8) return -1;
      if (box.type === path[0]) {
        return path.length === 1 ? p + box.hdr : mp4Find(b, p + box.hdr, p + box.size, path.slice(1));
      }
      p += box.size;
    }
    return -1;
  }

  function mp4Timescale(init) {
    const moov = mp4Find(init, 0, init.length, ["moov"]);
    const mdhd = moov < 0 ? -1 : mp4Find(init, moov, init.length, ["trak", "mdia", "mdhd"]);
    if (mdhd < 0) return 0;
    return readUint(init, mdhd + (init[mdhd] === 1 ? 20 : 12), 4);
  }

  function toBytes(data) {
    if (data instanceof ArrayBuffer) return new Uint8Array(data.slice(0));
    return new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  }

  function bufferedEnd(sb) {
    try { return sb.buffered.length ? sb.buffered.end(sb.buffered.length - 1) : 0; } catch (_) { return 0; }
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
      this.chunks = [];   // {id, start, end, audioStart, state: queued|sending|done|error, bytes, mime, buffer}
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

  // One per audio SourceBuffer. Mirrors the byte stream the SourceBuffer's parser sees
  // (only appends that completed), cuts it into units and indexes them by media time.
  class Tracker {
    constructor(sb, mime, session) {
      this.sb = sb;
      this.session = session;
      this.setMime(mime);
      this.units = [];      // sorted by t: {t, end, bytes, init, sent}
      this.byKey = new Map();
      this.lastUnit = 0;    // performance.now() when the last new unit arrived
      // abort() resets the SourceBuffer's parser, so ours restarts too.
      sb.addEventListener("abort", () => { this.buf = new Uint8Array(0); });
    }

    setMime(mime) {
      this.mime = mime;
      this.isMp4 = /mp4/i.test(mime);
      this.buf = new Uint8Array(0);
      this.init = null;
      this.initId = 0;
      this.scale = 0;
    }

    // Returns a cancel function for when the append is rejected (e.g. QuotaExceededError
    // when the buffer is full; YouTube then evicts and retries). The bytes only join our
    // copy of the stream once the SourceBuffer has actually accepted them.
    onAppend(data) {
      if (!settings.enabled) return null;
      const bytes = toBytes(data);
      const offset = this.sb.timestampOffset || 0;
      let aborted = false;
      const onAbort = () => { aborted = true; };
      const cleanup = () => {
        this.sb.removeEventListener("updateend", done);
        this.sb.removeEventListener("abort", onAbort);
      };
      const done = () => {
        cleanup();
        if (aborted) return;
        try { this.feed(bytes, offset); } catch (e) { log("capture error", e); this.buf = new Uint8Array(0); }
      };
      this.sb.addEventListener("updateend", done);
      this.sb.addEventListener("abort", onAbort);
      return cleanup;
    }

    feed(bytes, offset, final = false) {
      const rest = this.buf;
      const b = new Uint8Array(rest.length + bytes.length);
      b.set(rest, 0);
      b.set(bytes, rest.length);
      const used = this.isMp4 ? this.parseMp4(b, offset, final) : this.parseWebm(b, offset, final);
      this.buf = b.slice(used);
    }

    setInit(bytes) {
      const same = this.init && this.init.length === bytes.length && this.init.every((x, i) => x === bytes[i]);
      if (same) return;
      this.init = bytes;
      this.initId++;
      this.scale = this.isMp4 ? mp4Timescale(bytes) : webmTimecodeScale(bytes);
    }

    addUnit(t, end, bytes) {
      const key = `${this.initId}:${t.toFixed(3)}`;
      const old = this.byKey.get(key);
      if (old && (old.sent || old.bytes)) return; // re-append of a unit we already have
      const u = { t, end: Math.max(end, t + 0.001), bytes, init: this.init, initId: this.initId, sent: false };
      this.byKey.set(key, u);
      if (old) this.units[this.units.indexOf(old)] = u;
      else {
        let i = this.units.length;
        while (i > 0 && this.units[i - 1].t > t) i--;
        this.units.splice(i, 0, u);
      }
      this.lastUnit = performance.now();
    }

    parseWebm(b, offset, final) {
      let p = 0;
      while (p < b.length) {
        const id = readVint(b, p, true), sz = id && readVint(b, p + id.len, false);
        if (!sz) break;
        if (id.val === EBML_ID) {
          const n = webmInitLength(b, p, final);
          if (n < 0) break;
          this.setInit(b.slice(p, p + n));
          p += n;
        } else if (id.val === CLUSTER_ID) {
          const end = webmClusterEnd(b, p, final);
          if (end < 0) break;
          if (this.init) {
            const { tc, maxRel } = webmClusterTimes(b, p, end);
            if (tc !== null) {
              const s = this.scale / 1e9;
              // +20 ms: the last block's own duration (typical Opus frame)
              this.addUnit(tc * s + offset, (tc + maxRel) * s + 0.02 + offset, b.slice(p, end));
            }
          }
          p = end;
        } else if (id.val === SEGMENT_ID) {
          p += id.len + sz.len; // step into the Segment
        } else {
          if (sz.unknown) { p += id.len + sz.len; continue; }
          const end = p + id.len + sz.len + sz.val;
          if (end > b.length) break;
          p = end; // Cues, Tags, Void, ...
        }
      }
      return p;
    }

    parseMp4(b, offset, final) {
      let p = 0;
      while (p < b.length) {
        const box = mp4Box(b, p);
        if (!box || box.size < 8) break;
        if (box.type === "ftyp" || box.type === "moov") {
          // Init segment: ftyp (+ anything) + moov.
          let q = p, moovEnd = -1;
          while (q < b.length) {
            const x = mp4Box(b, q);
            if (!x || q + x.size > b.length) break;
            q += x.size;
            if (x.type === "moov") { moovEnd = q; break; }
          }
          if (moovEnd < 0) break;
          this.setInit(b.slice(p, moovEnd));
          p = moovEnd;
        } else if (box.type === "moof") {
          const moofEnd = p + box.size;
          const mdat = moofEnd <= b.length ? mp4Box(b, moofEnd) : null;
          if (!mdat || moofEnd + mdat.size > b.length) break;
          const end = moofEnd + mdat.size;
          const tfdt = mp4Find(b, p + box.hdr, moofEnd, ["traf", "tfdt"]);
          if (this.init && this.scale && tfdt >= 0) {
            const base = b[tfdt] === 1 ? readUint(b, tfdt + 4, 8) : readUint(b, tfdt + 4, 4);
            const t = base / this.scale + offset;
            // Fragment end: the start of the next one (fixed up below), else what's buffered.
            const prev = this.units[this.units.length - 1];
            if (prev && prev.initId === this.initId && t > prev.t && t - prev.end < 30) prev.end = t;
            const be = bufferedEnd(this.sb);
            this.addUnit(t, be > t ? be : t + 1, b.slice(p, end));
          }
          p = end;
        } else {
          if (p + box.size > b.length) break;
          p += box.size; // styp, sidx, emsg, free, prft, stray mdat
        }
      }
      return p;
    }

    // Stream ended: the last unknown-size cluster is complete now.
    finalize() {
      if (this.buf.length) this.feed(new Uint8Array(0), this.sb.timestampOffset || 0, true);
    }

    // The unsent, contiguous run of units starting at or after time t.
    runFrom(t) {
      const u = this.units;
      let i = u.findIndex((x) => !x.sent && x.bytes && x.end > t - 0.25);
      if (i < 0) return null;
      let j = i + 1;
      while (j < u.length && !u[j].sent && u[j].bytes && u[j].initId === u[i].initId &&
             u[j].t - u[j - 1].end < 0.15 && u[j].t > u[j - 1].t) j++;
      return u.slice(i, j);
    }

    // Emit these units as one chunk.
    emit(run, reason) {
      for (const x of run) x.sent = true;
      const start = run[0].t, end = run[run.length - 1].end;
      const bytesOf = run.map((x) => x.bytes);
      for (const x of run) x.bytes = null; // the chunk holds them now
      if (end - start < 0.05 || this.session.covered(start, end)) return;
      const init = run[0].init;
      let len = init.length;
      for (const x of bytesOf) len += x.length;
      const buf = new Uint8Array(len);
      buf.set(init, 0);
      let off = init.length;
      for (const x of bytesOf) { buf.set(x, off); off += x.length; }
      this.session.chunks.push({
        id: `${this.session.id}-${start.toFixed(2)}`,
        start, end, audioStart: start, state: "queued", bytes: buf, mime: this.mime, buffer: null, retryAt: 0,
      });
      log(`chunk ${start.toFixed(1)}-${end.toFixed(1)}s (${run.length} units, ${(len / 1024) | 0} KiB) queued: ${reason}`);
      pump();
    }

    // Called every tick: decide whether the audio ahead of the playhead should be sent.
    maybeIdleFlush(video) {
      if (!video) return;
      const t = video.currentTime;
      for (let guard = 0; guard < 20; guard++) {
        const run = this.runFrom(t);
        if (!run) return;
        const start = run[0].t, dur = run[run.length - 1].end - start;
        const first = !this.session.covered(start - 0.5, start);
        const target = first ? Math.min(settings.firstChunkSeconds, settings.chunkSeconds) : settings.chunkSeconds;
        if (dur >= target) {
          // Full chunk: take units up to the target length.
          let k = 1;
          while (k < run.length && run[k].end - start <= target) k++;
          this.emit(run.slice(0, k), `reached ${target}s`);
          continue;
        }
        const idle = performance.now() - this.lastUnit;
        const ahead = start - t; // how soon playback reaches this audio
        // Send early enough that processing finishes before playback gets there.
        const lead = Math.min(30, Math.max(4, 1.5 * procSeconds + 2));
        let reason = null;
        if (this.session.ended && run[run.length - 1] === this.units[this.units.length - 1]) reason = "end of video";
        else if (ahead <= lead && idle > 1000 && dur >= 3) reason = `playback reaches it in ${Math.max(0, ahead).toFixed(1)}s`;
        else if (ahead <= 0.5 && idle > 4000 && dur > 0.2) reason = "playback waiting on it";
        else if (ahead <= 2 && idle > 10000 && dur > 0.2) reason = "YouTube stopped buffering";
        if (reason) this.emit(run, reason);
        return;
      }
    }

    // Unsent audio that covers time t (captured but not sent yet)?
    covers(t) {
      return this.units.some((x) => !x.sent && x.bytes && x.t <= t + 0.5 && x.end > t);
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
    if (isMs && !disposed) { const s = sessionFor(obj); s.url = url; sessions.set(url, s); }
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
      if (!disposed && /^audio\//i.test(mime)) trackers.set(sb, new Tracker(sb, mime, sessionFor(this)));
      return sb;
    };
    const origEos = proto.endOfStream;
    proto.endOfStream = function () {
      const s = sessionByMs.get(this);
      if (s && !disposed) {
        s.ended = true;
        for (const tr of trackersOf(s)) { try { tr.finalize(); } catch (e) { log("capture error", e); } }
      }
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
        if (t) t.setMime(mime);
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
      let src;
      try {
        src = c.createMediaElementSource(video);
      } catch (e) {
        // Already connected by something we can't take over (e.g. an old copy of this script).
        needsReload = true;
        return null;
      }
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
    return trackersOf(session).some((tr) => tr.covers(t));
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
    else if (needsReload) text = "Music remover: reload the page to remove music from this video";
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
