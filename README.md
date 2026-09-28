# YouTube Music Remover

Removes background music (instruments) from YouTube videos **while you watch**, and keeps
speech and non-music sound effects.

It has two parts:

* **`extension/`**: a Chrome/Edge extension (Manifest V3). It copies the audio that YouTube's
  player has already downloaded into its buffer, cuts it into ~1 minute chunks, sends them
  to the local server, and plays the returned music-free audio in sync with the video.
* **`server/`**: a small local Python server that runs a source-separation model and returns
  everything except the music. By default it uses **BandIt Plus**, a *cinematic* separation
  model that splits audio into speech / music / sound effects. The server returns
  speech + effects. It runs on your GPU if you have one (NVIDIA CUDA or Apple Metal), and
  the popup has a **Force CPU** switch. See [`bench/`](bench/README.md) for how the available
  removers compare.

Everything runs on your machine. No audio leaves your computer.

## How it works

```
YouTube player ──appendBuffer(audio segment)──► MediaSource buffer   (the "browser cache")
        │  inject.js hooks SourceBuffer.appendBuffer and copies each audio segment,
        │  noting which media time range it covers (from SourceBuffer.buffered)
        ▼
 chunker: init segment + contiguous segments ≈ 60 s  →  a standalone .webm / .mp4 file
        │  bridge.js → background.js → POST http://127.0.0.1:8765/separate
        ▼
 server.py: ffmpeg decode → BandIt Plus → keep speech + effects → Opus
        ▼
 inject.js: decodes the chunk and plays it via WebAudio, locked to video.currentTime;
            the <video>'s own audio is routed through a gain node that is set to 0
```

* Reading the buffer (instead of downloading the video again) means no extra network traffic, and it
  still works with YouTube's newer SABR/UMP streaming, because every byte eventually goes
  through `SourceBuffer.appendBuffer`.
* The chunk is cut only at segment/cluster boundaries, so each chunk decodes on its own.
* The first chunk after a start or seek is shorter (20 s by default), so playback starts sooner.
  After that, chunks are 60 s. If YouTube stops buffering further ahead, whatever is buffered
  gets sent early.
* The next chunk is scheduled sample-accurately so chunk boundaries don't click. Drift over 80 ms
  (after a seek, a playback speed change, and so on) triggers a resync.
* A small badge on the player shows the status. When a video was held while its audio was
  processed, a short chime plays just before it starts.

## Setup

### 1. Server

Requires Python 3.9+, `git`, and `ffmpeg` (with libopus) on your PATH.

```bash
cd server
python -m venv .venv && source .venv/bin/activate    # optional
# NVIDIA GPU: install the CUDA build of torch + torchaudio first (https://pytorch.org/get-started/locally/)
pip install -r requirements.txt
python server.py
```

On first use it downloads the model code (ZFTurbo's MIT-licensed
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
pinned to a specific commit) and the weights from GitHub releases into `server/.cache/`:
BandIt Plus is 149 MB, and BS-RoFormer (only needed for the other two removers) is 640 MB.

**GPU / CPU:** the server uses an NVIDIA (CUDA) or Apple Silicon (MPS) GPU automatically when
PyTorch can see one. The popup shows which GPU was found. Tick **Force CPU** to run on the
CPU instead; the model moves over on the next chunk, with no restart needed.

**Removers** (chosen in the popup):

| Remover | Keeps | Notes |
|---|---|---|
| `bandit` (default) | speech + sound effects | BandIt Plus (DnR). Singing mostly goes out with the music. |
| `hybrid` | speech + effects + singing | Runs BS-RoFormer on BandIt's music stem to put sung vocals back. Slower. |
| `vocals` | voices only | BS-RoFormer. Cleanest speech, but sound effects are removed too. |

Environment variables: `MR_ENGINE` (default remover), `MR_DEVICE` (pin a device and ignore the
popup), `MR_PORT` (default `8765`), `MR_OVERLAP` (window overlap: default 4 on GPU, 2 on CPU;
lower is faster).

**Speed:** these models are heavy. On a recent NVIDIA GPU a 60 s chunk takes a few seconds. On
CPU, BandIt runs slower than real time (≈4× real time on a 4-core laptop-class CPU),
so with **Pause until ready** you will wait between chunks. A GPU is strongly recommended.

### 2. Extension

1. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked**, and pick the `extension/` folder.
2. Open (or reload) a YouTube video.

In the toolbar popup you can:

* turn it on or off
* choose what happens when playback reaches audio that isn't processed yet:
  * **Pause until ready** (default): you never hear the original music
  * **Muted**
  * **Original audio**
* force the CPU, pick the remover, and turn the "ready" chime on or off
* change the chunk length and the server URL

## Limitations

* BandIt Plus was trained on film-style mixes (English audiobook speech, general music,
  Freesound effects). Sounds that are part of the music, and music-like effects (sirens,
  bells), can go either way.
* DRM-protected videos (Premium movies and similar) use encrypted buffers and can't be processed.
* Playback faster or slower than 1× changes the pitch of the processed audio.
* The extension only hooks the desktop/mobile web player on `youtube.com` (not embeds on other sites).

## Development

`inject.js` runs in the page's main world at `document_start`, so it can patch
`MediaSource`/`SourceBuffer` before YouTube's player loads. `bridge.js` (isolated world) relays
messages to `background.js`, which makes the localhost request. Doing that from the
service worker avoids YouTube's CSP and the browser's local-network restrictions.
