# YouTube Music Remover

Removes background music from YouTube videos **while you watch**, and keeps the speech.

It has two parts:

* **`extension/`**: a Chrome/Edge extension (Manifest V3). It copies the audio that YouTube's
  player has already downloaded into its buffer, cuts it into ~1 minute chunks, sends them
  to the local server, and plays the returned music-free audio in sync with the video.
* **`server/`**: a small local Python server that runs [Demucs](https://github.com/facebookresearch/demucs),
  a standard music source-separation model, and returns only the *vocals* stem.

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
 server.py: ffmpeg decode → Demucs → keep "vocals" → Opus
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
* A small badge on the player shows the status.

## Setup

### 1. Server

Requires Python 3.9+ and `ffmpeg` (with libopus) on your PATH.

```bash
cd server
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt                       # for NVIDIA GPU, install the CUDA build of torch first
python server.py
```

The first run downloads the model (~80 MB). Options (environment variables):

| Variable    | Default    | Meaning                                                     |
|-------------|------------|-------------------------------------------------------------|
| `MR_MODEL`  | `htdemucs` | `htdemucs_ft` gives better quality but is about 4× slower   |
| `MR_DEVICE` | auto       | `cuda`, `mps` (Apple Silicon) or `cpu`                      |
| `MR_PORT`   | `8765`     |                                                             |

Speed: on a modern GPU, a 60 s chunk takes a few seconds. On CPU it takes roughly 0.3–1× real time,
which still keeps up but gives a longer wait at the start.

### 2. Extension

1. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked**, and pick the `extension/` folder.
2. Open (or reload) a YouTube video.

In the toolbar popup you can:

* turn it on or off
* choose what happens when playback reaches audio that isn't processed yet:
  * **Pause until ready** (default): you never hear the original music
  * **Muted**
  * **Original audio**
* change the chunk length and the server URL

## Limitations

* **Singing counts as vocals.** Demucs separates voice from instruments, so a sung vocal line
  stays in. Speech, which is what most videos contain, comes through clean.
* DRM-protected videos (Premium movies and similar) use encrypted buffers and can't be processed.
* Playback faster or slower than 1× changes the pitch of the processed audio.
* The extension only hooks the desktop/mobile web player on `youtube.com` (not embeds on other sites).

## Development

`inject.js` runs in the page's main world at `document_start`, so it can patch
`MediaSource`/`SourceBuffer` before YouTube's player loads. `bridge.js` (isolated world) relays
messages to `background.js`, which makes the localhost request. Doing that from the
service worker avoids YouTube's CSP and the browser's local-network restrictions.
