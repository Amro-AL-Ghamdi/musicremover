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
  speech + effects. It runs on your GPU if you have one (NVIDIA CUDA, AMD ROCm or Apple Metal), and
  the popup has a **Force CPU** switch. [`bench/`](bench/README.md) has the measurements
  behind these choices.

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
 server.py: ffmpeg decode → BandIt Plus → keep speech + effects → bleed filter → Opus
        ▼
 inject.js: decodes the chunk and plays it via WebAudio, locked to video.currentTime;
            the <video>'s own audio is routed through a gain node that is set to 0
```

* Reading the buffer (instead of downloading the video again) means no extra network traffic, and it
  still works with YouTube's newer SABR/UMP streaming, because every byte eventually goes
  through `SourceBuffer.appendBuffer`.
* The chunk is cut only at segment/cluster boundaries, so each chunk decodes on its own.
* The first chunk after a start or seek is shorter (20 s by default), so playback starts sooner.
  After that, chunks are up to 60 s. YouTube often buffers only ~10 s ahead, so a shorter chunk
  is sent as soon as playback is close enough that processing would otherwise finish too late
  (based on how long recent chunks took).
* Capture reads timing from the audio data itself: the appended byte stream is rebuilt exactly
  as the player's parser sees it, split into complete WebM clusters / MP4 fragments, and each
  is indexed by its own timestamp. Chunks are runs of consecutive units, so it doesn't matter
  how YouTube slices, repeats or paces its appends.
* If playback waits more than 20 s on a spot that nothing is being captured or processed for,
  it continues muted instead of hanging.
* The next chunk is scheduled sample-accurately so chunk boundaries don't click. Drift over 80 ms
  (after a seek, a playback speed change, and so on) triggers a resync.
* A thin strip on YouTube's progress bar shows the processed parts (green), the part being
  processed (amber) and parts waiting to be sent (grey). Next to the time it says how far the
  music has been removed, e.g. "· music removed to 2:35". On Shorts the strip is at the bottom
  of the video.
* A small badge on the player shows the status. When a video was held while its audio was
  processed, a short chime plays just before it starts.
* Works on regular videos and Shorts. Tabs that were already open when the extension was
  installed or reloaded are picked up automatically: the playing video restarts at the same
  position so its audio can be captured.

## Setup

### 1. Server

Requires Python 3.9+, `git`, and `ffmpeg` (with libopus) on your PATH.

```bash
cd server
python -m venv .venv && source .venv/bin/activate    # optional
python install.py      # finds your GPU and installs the matching PyTorch + everything else
python server.py
```

`install.py` picks the PyTorch build for your hardware: CUDA 13.0 or 12.6 for NVIDIA (based on
your driver version), ROCm for AMD, Metal for Apple Silicon, or the small CPU-only build.
It then installs `requirements.txt`. Use `--dry-run` to see the commands first, or
`--target cpu|nvidia|amd|apple` to override the detection. To install by hand instead,
follow the table below, then run `pip install -r requirements.txt`.

On first use it downloads the model code (ZFTurbo's MIT-licensed
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
pinned to a specific commit) and the weights of the model you pick into `server/.cache/`:
BandIt Plus 149 MB, Demucs 168 MB, UVR Voc FT 67 MB and MelBand RoFormer 913 MB (all from
GitHub releases), DnR Demucs from Zenodo.

**GPU / CPU:** the server uses an NVIDIA, AMD or Apple Silicon GPU automatically when
PyTorch can see one. The popup shows which GPU was found. Tick **Force CPU** to run on the
CPU instead; the model moves over on the next chunk, with no restart needed. If you have a GPU
but installed a CPU-only PyTorch, the server log and the popup say so and tell you what to install.

#### GPU setup

The GPU is picked up through PyTorch. `install.py` does this for you; by hand, install the
matching build before `requirements.txt`:

| GPU | OS | Install |
|---|---|---|
| NVIDIA | Windows / Linux | CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) |
| AMD Radeon | Linux | ROCm build from [pytorch.org](https://pytorch.org/get-started/locally/) (choose *ROCm*), or AMD's wheels ([guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)) |
| AMD Radeon RX 7000 / 9000, Ryzen AI 300 / Max | Windows | AMD's ROCm PyTorch for Windows (public preview; needs Python 3.12 and a recent Adrenalin driver): [guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html). WSL2 also works: [guide](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-pytorch.html) |
| Apple Silicon | macOS | the regular `pip install torch torchaudio` (Metal is built in) |

AMD notes:
* ROCm shows up in PyTorch as the `cuda` device, so nothing else needs configuring. The popup
  shows e.g. `Radeon RX 7900 XTX (AMD ROCm)`.
* The server sets `MIOPEN_FIND_MODE=FAST` so the first chunk doesn't stall for minutes
  while ROCm benchmarks kernels.
* Linux, older or unlisted Radeon cards (e.g. RX 6000 series): if ROCm doesn't recognize the
  card, start the server with `HSA_OVERRIDE_GFX_VERSION=10.3.0` (RDNA2) or `11.0.0` (RDNA3).
* Older Radeons on Windows (RX 6000 and earlier) aren't supported by ROCm for Windows. Use the
  CPU, or run the server under Linux or WSL2.

**Models** (popup: Model). All five run on the same pipeline and bleed filter. Numbers are at
the default (Normal) bleed suppression:

| Model | Keeps | Music left under speech | Music in pauses | Sound effects | CPU time per audio second |
|---|---|---|---|---|---|
| **BandIt Plus** (default) | speech + sound effects | −20.6 dB | silent | kept (−1.3 dB) | ≈6 s |
| Demucs | voices only | −24.9 dB | −73 dB | **removed** (−17 dB) | ≈1.4 s |
| UVR Voc FT | voices only | −26.9 dB | −69 dB | **removed** (−25 dB) | ≈2.3 s |
| MelBand RoFormer | voices only | **−28.1 dB** | silent | **removed** (−29 dB) | ≈11 s |
| DnR Demucs (experimental) | speech + sound effects | not measured | not measured | kept (by design) | not measured |

* **BandIt Plus** is a *cinematic* model: it separates dialogue, music and sound effects, so
  effects survive.
* **Demucs** (HTDemucs fine-tuned for vocals) is about 4× faster and leaves less music under
  speech. It sorts audio into "vocals" and "everything else", though, so sound effects go with
  the music. Pick it when you only care about voices.
* **DnR Demucs** is the Hybrid Demucs baseline from the BandIt paper, trained on the same
  dialogue/music/effects data as BandIt. Its weights are on Zenodo (CC-BY-NC 4.0), which our
  test machine couldn't reach, so it is untested. If the automatic download fails, the popup
  badge says where to put `dnr-demucs.ckpt` by hand.

* **UVR Voc FT** (UVR-MDX-NET-Voc_FT) is the model several music-muting tools use. The ONNX
  file is converted to PyTorch with `onnx2torch`, so it runs on any GPU PyTorch supports
  (including AMD). Its output matches audio-separator's to 27 dB after volume matching;
  audio-separator also scales its output by the input's peak level, which we don't.
* **MelBand RoFormer** (Kim's vocal model, fine-tuned by unwa) is a newer model from the same
  UVR community. It leaves the least music of all models here, but it is the slowest.

The voice-only models (Demucs, Voc FT, MelBand) remove sound effects together with the music.
Only BandIt Plus and DnR Demucs keep them. See [`bench/`](bench/README.md) for all
measurements, including models that were tried and dropped.

**Bleed suppression** (popup: Off / Normal / Strong): after the model, an extra spectral mask
uses the model's own music estimate to push leftover music down further. The strength behind
each level depends on the model (BandIt 1/4, Demucs and Voc FT 16/64, MelBand 4/16), because
each model stops improving at a different point.
Measured on test mixes with known ground truth:

| Model, level | Music left | Music in pauses | Sound effects kept | Speech kept |
|---|---|---|---|---|
| BandIt Plus, Off | −19.5 dB | −103 dB | −1.1 dB | −0.8 dB |
| **BandIt Plus, Normal** (1) | −20.6 dB | silent | −1.3 dB | −1.0 dB |
| BandIt Plus, Strong (4) | −21.2 dB | silent | −1.4 dB | −1.2 dB |
| Demucs, Off | −22.1 dB | −54 dB | −14.4 dB | −0.1 dB |
| **Demucs, Normal** (16) | −24.9 dB | −73 dB | −17.4 dB | −0.8 dB |
| Demucs, Strong (64) | −25.7 dB | −79 dB | −18.9 dB | −1.3 dB |
| Voc FT, Off | −24.8 dB | −47 dB | −20.4 dB | −0.2 dB |
| **Voc FT, Normal** (16) | −26.9 dB | −69 dB | −25.0 dB | −1.0 dB |
| Voc FT, Strong (64) | −27.3 dB | −80 dB | −26.5 dB | −1.6 dB |
| MelBand, Off | −26.9 dB | −116 dB | −27.6 dB | −0.2 dB |
| **MelBand, Normal** (4) | −28.1 dB | silent | −29.2 dB | −0.6 dB |
| MelBand, Strong (16) | −28.0 dB | silent | −30.3 dB | −1.1 dB |

The filter costs almost nothing to run. Past these levels it stops helping: the remaining
bleed is music the model itself mistakes for speech or effects. Running a model a second time
on its own output was also tested and didn't lower the bleed.

**Speed:** on a GPU the model runs in half precision (fp16) with batches of 8 windows. If fp16
fails or gives bad output on your card, it switches to full precision automatically and says so in the
server log. The first chunk after starting the server (or switching models) is slow while the
model loads. On CPU, BandIt runs slower than real time (≈6 s per second of audio on a 4-core
CPU), so with **Pause until ready** you will wait between chunks; Demucs is about 4× faster.
A GPU is strongly recommended.

Environment variables: `MR_DEVICE` (pin a device and ignore the popup), `MR_PORT` (default
`8765`), `MR_FP16=0` (disable half precision), `MR_OVERLAP` (window overlap, default 2),
`MR_BATCH` (windows per batch, default 8 on GPU, 4 on CPU).

### 2. Extension

1. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked**, and pick the `extension/` folder.
2. Open a YouTube video or Short. Already-open YouTube tabs are picked up automatically.

In the toolbar popup you can:

* turn it on or off
* choose what happens when playback reaches audio that isn't processed yet:
  * **Pause until ready** (default): you never hear the original music
  * **Muted**
  * **Original audio**
* pick the model, force the CPU, set the bleed suppression, and turn the "ready" chime on or off
* change the chunk length and the server URL

## Limitations

* Some music bleed remains under speech: about −20 dB with BandIt (roughly 1/10 of the original
  loudness), −25 dB with Demucs. BandIt usually treats singing as music and removes it;
  Demucs keeps it.
* BandIt Plus was trained on film-style mixes (English audiobook speech, general music,
  Freesound effects). Sounds that are part of the music, and music-like effects (sirens,
  bells), can go either way.
* DRM-protected videos (Premium movies and similar) use encrypted buffers and can't be processed.
* Playback faster or slower than 1× changes the pitch of the processed audio.
* The extension only hooks the player on `youtube.com` (videos and Shorts), not embeds on other
  sites.

## Development

`inject.js` runs in the page's main world at `document_start`, so it can patch
`MediaSource`/`SourceBuffer` before YouTube's player loads. `bridge.js` (isolated world) relays
messages to `background.js`, which makes the localhost request. Doing that from the
service worker avoids YouTube's CSP and the browser's local-network restrictions.
