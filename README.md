# YouTube Music Remover

Removes background music (instruments) from YouTube videos **while you watch**, and keeps
speech and non-music sound effects.

It has two parts:

* **`extension/`**: a Chrome/Edge extension (Manifest V3). It copies the audio that YouTube's
  player has already downloaded into its buffer, cuts it into ~1 minute chunks, sends them
  to the local server, and plays the returned music-free audio in sync with the video.
* **`server/`**: a small local Python server that runs a source-separation model and returns
  everything except the music. By default it combines two models: **UVR Voc FT** for the
  voices and **DnR Demucs**, a *cinematic* model that splits audio into speech / music /
  sound effects, for the effects. The server returns voices + effects. It runs on your GPU if you have one (NVIDIA CUDA, AMD ROCm or Apple Metal), and
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
 server.py: ffmpeg decode → Voc FT + DnR Demucs → keep voices + effects → bleed filter → Opus
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

## Quick start

### Download the app (no Python needed)

1. **Download** from the [latest release](https://github.com/siba1426/musicremover/releases/latest):
   * Windows: **`MusicRemover-Setup.exe`**. Run it; no admin rights needed. Windows may warn
     that the installer isn't signed: click **More info → Run anyway**.
   * Linux: **`MusicRemover-x86_64.AppImage`**. Make it executable (`chmod +x`, or
     Properties → Permissions) and double-click it. Or use `MusicRemover-linux-x86_64.tar.gz`:
     unpack it and run `./run.sh`.
2. **Start Music Remover** (Start menu / desktop on Windows, the AppImage on Linux). The first
   start finds your GPU and downloads the matching PyTorch (1–3 GB, or ~200 MB for the CPU
   build) and the models. Later starts take a few seconds. Leave the window open while you
   watch; closing it stops the server. It can also be stopped with **Stop Music Remover**
   (Windows Start menu), `./MusicRemover-x86_64.AppImage --stop` or `./stop.sh` (Linux).
3. **Load the extension** once: the window prints the extension folder
   (`%LOCALAPPDATA%\MusicRemover\extension` on Windows, `~/.local/share/musicremover/extension`
   on Linux). Open `chrome://extensions` (or `edge://extensions`), turn on **Developer mode**,
   click **Load unpacked** and pick that folder.
4. Open a YouTube video. Already-open YouTube tabs are picked up automatically.

The app contains Python and every dependency except PyTorch, which depends on your GPU.
PyTorch, the models and the extension copy go to the folder above; uninstalling on Windows
removes it, on Linux delete it yourself. After installing a newer version, reload the
extension in `chrome://extensions`.

### Or install from source

1. **Download** this repository: `git clone https://github.com/siba1426/musicremover.git`, or on
   GitHub click **Code → Download ZIP** and unzip it.
2. **Install** (needs [Python](https://www.python.org/downloads/) 3.9+; on Windows tick
   "Add python.exe to PATH", and pick 3.12 if you have an AMD GPU):
   * Windows: double-click **`install.bat`**
   * Linux / macOS: run **`./install.sh`**

   It finds your GPU (NVIDIA, AMD, Apple Silicon) and installs the matching PyTorch, or the
   small CPU-only build if there's no usable GPU. It also installs everything else and downloads
   the default models, then starts the server. Nothing else is needed: ffmpeg comes bundled if
   you don't have it.
3. **Next time, start the server** with **`run.bat`** (Windows) or **`./run.sh`**, and leave it
   running while you watch. To stop it (also when it runs in the background or is still
   installing), use **`stop.bat`** or **`./stop.sh`**.
4. **Load the extension** once: open `chrome://extensions` (or `edge://extensions`), turn on
   **Developer mode**, click **Load unpacked** and pick the `extension/` folder.
5. Open a YouTube video. Already-open YouTube tabs are picked up automatically.

After pulling an update: reload the extension in `chrome://extensions` and restart the
server. If the update changed the Python requirements, `run.bat` / `run.sh` installs them
automatically before starting.

## Setup (details)

### 1. Server

The install scripts above wrap these steps. By hand (Python 3.9+):

```bash
cd server
python -m venv .venv && source .venv/bin/activate    # optional
python install.py      # finds your GPU and installs the matching PyTorch + everything else
python server.py
```

ffmpeg is taken from your PATH if installed, otherwise from the bundled `imageio-ffmpeg`
package.

`install.py` picks the PyTorch build for your hardware: CUDA 13.0 or 12.6 for NVIDIA (based on
your driver version), ROCm for AMD, Metal for Apple Silicon, or the small CPU-only build.
On Linux with an AMD card it reads the GPU's gfx target (e.g. `gfx1200` for an RX 9060 XT) and
first tries AMD's build for just that GPU family (`gfx120X-all` for RX 9000, `gfx110X-all` for
RX 7000, `gfx1150`/`gfx1151` for Ryzen AI), which leaves out the ROCm libraries for every other
GPU; if that fails it falls back to the general ROCm build from pytorch.org.
It then installs `requirements.txt`. While installing, temporary files go to
`server/.cache/tmp` on disk instead of the system temp folder (`/tmp` is often in RAM and too
small for the PyTorch wheels); the folder is deleted afterwards. Use `--dry-run` to see the commands first, or
`--target cpu|nvidia|amd|apple` to override the detection. To install by hand instead,
follow the table below, then run `pip install -r requirements.txt`.

The installer downloads the model weights into `server/.cache/`: UVR Voc FT (67 MB, from UVR's
GitHub releases) and DnR Demucs (from Zenodo). The default Voc FT + DnR Demucs option uses both files.

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
| Apple Silicon | macOS | the regular `pip install torch torchaudio torchvision` (Metal is built in) |

AMD notes:
* ROCm shows up in PyTorch as the `cuda` device, so nothing else needs configuring. The popup
  shows e.g. `Radeon RX 7900 XTX (AMD ROCm)`.
* The server sets `MIOPEN_FIND_MODE=FAST` so the first chunk doesn't stall for minutes
  while ROCm benchmarks kernels.
* Linux, older or unlisted Radeon cards (e.g. RX 6000 series): if ROCm doesn't recognize the
  card, start the server with `HSA_OVERRIDE_GFX_VERSION=10.3.0` (RDNA2) or `11.0.0` (RDNA3).
* Older Radeons on Windows (RX 6000 and earlier) aren't supported by ROCm for Windows. Use the
  CPU, or run the server under Linux or WSL2.

**Models** (popup: Model). All three run on the same pipeline and bleed filter. Numbers are at
the default (Normal) bleed suppression:

| Model | Keeps | Music left under speech | Music in pauses | Sound effects | CPU time per audio second |
|---|---|---|---|---|---|
| **Voc FT + DnR Demucs** (default) | voices + sound effects | −18.8 dB¹ | −48 dB¹ | **kept (−0.6 dB)** | ≈4 s (0.56 s on an RX 9060 XT) |
| UVR Voc FT | voices only | **−26.9 dB** | −69 dB | **removed** (−25 dB) | ≈2.3 s |
| UVR Voc FT int8 | voices only | −27.0 dB | −74 dB | **removed** (−26 dB) | **≈1.1 s** |
| DnR Demucs | speech + sound effects | −18.5 dB | −56 dB | kept (−1.5 dB) | ≈1.3 s |

¹ Measured with the earlier single filter. Each part now has its own filter (see below), which
should improve both; re-run the benchmark to get current numbers.

* **Voc FT + DnR Demucs** combines the two: voices from Voc FT, sound effects from DnR Demucs
  (Voc FT has no effects stem; its "other" is music and effects together). It keeps effects and
  speech best. Both models run on every chunk, and they're shared with the standalone options,
  so nothing loads twice. Each part is filtered by the model it came from: voices with Voc FT's
  own estimate (at 4× the strength), effects with DnR's music estimate. Measure it with
  `python bench/benchmark.py --engine voc_ft_dnr`.
* **UVR Voc FT** (UVR-MDX-NET-Voc_FT) is the model several music-muting tools use. It leaves the
  least music, but it sorts audio into "vocals" and "everything else", so sound effects go with
  the music. The ONNX file is converted to PyTorch with `onnx2torch`, so it runs on any GPU
  PyTorch supports (including AMD). Its output matches audio-separator's to 27 dB after volume
  matching; audio-separator also scales its output by the input's peak level, which we don't.
* **UVR Voc FT int8** is Voc FT for weaker PCs: on the CPU (no usable GPU, or **Force CPU**) its
  frequency layers run as 8-bit integers with onnxruntime, about 2× faster than Voc FT on the CPU
  and with less memory (≈3.5 GB). It scores the same as Voc FT, except speech is 0.4 dB softer.
  On a GPU it runs the regular Voc FT there in half precision, which is faster still. The
  quantized file is made from the Voc FT download on first use, in a few seconds.
* **DnR Demucs** is the Hybrid Demucs baseline from the BandIt paper, trained on
  dialogue/music/effects mixes (DnR). Its weights are on Zenodo (CC-BY-NC 4.0). If the automatic
  download fails, the popup badge says where to put `dnr-demucs.ckpt` by hand.

See [`bench/`](bench/README.md) for all measurements, including models that were tried and
dropped (BandIt Plus, HTDemucs, MelBand RoFormer and others).

**Bleed suppression** (popup: Off / Normal / Strong): after the model, an extra spectral mask
uses the model's own music estimate to push leftover music down further. The strength behind
each level depends on the model (Voc FT + DnR 4/16 for effects and 16/64 for voices, Voc FT
and Voc FT int8 16/64, DnR Demucs 1/4), because each model stops improving
at a different point.
Measured on test mixes with known ground truth:

| Model, level | Music left | Music in pauses | Sound effects kept | Speech kept |
|---|---|---|---|---|
| Voc FT, Off | −24.8 dB | −47 dB | −20.4 dB | −0.2 dB |
| **Voc FT, Normal** (16) | −26.9 dB | −69 dB | −25.0 dB | −1.0 dB |
| Voc FT, Strong (64) | −27.3 dB | −80 dB | −26.5 dB | −1.6 dB |
| Voc FT int8, Off | −25.2 dB | −48 dB | −19.1 dB | −0.4 dB |
| **Voc FT int8, Normal** (16) | −27.0 dB | −74 dB | −25.8 dB | −1.4 dB |
| Voc FT int8, Strong (64) | −27.4 dB | −86 dB | −27.4 dB | −2.1 dB |

The filter costs almost nothing to run. Past these levels it stops helping: the remaining
bleed is music the model itself mistakes for speech or effects. Running a model a second time
on its own output was also tested and didn't lower the bleed.

**Speed:** on a GPU the model runs in half precision (fp16) with batches of 8 windows. If fp16
fails or gives bad output on your card, it switches to full precision automatically and says so in the
server log. The first chunk after starting the server (or switching models) is slow while the
model loads. On CPU, the default runs slower than real time (≈4 s per second of audio on a
4-core CPU), so with **Pause until ready** you will wait between chunks; Voc FT or DnR Demucs
alone are faster.
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
* turn the cache of recently watched videos on or off, or clear it: processed audio of the last
  10 videos is kept (≈1 MB per minute), so rewatching them, or seeking back, plays it straight
  away; only parts that weren't processed yet go to the server. It's kept per model and bleed
  setting, and dropped when the video's length changes
* change the chunk length and the server URL

## Limitations

* Some music bleed remains under speech: about −19 dB with the default (roughly 1/9 of the
  original loudness), −27 dB with Voc FT alone. Voc FT keeps singing as voice.
* DnR Demucs was trained on film-style mixes (English audiobook speech, general music,
  Freesound effects). Sounds that are part of the music, and music-like effects (sirens,
  bells), can go either way.
* DRM-protected videos (Premium movies and similar) use encrypted buffers and can't be processed.
* At speeds other than 1× the processed audio is time-stretched (WSOLA) to keep its pitch, like
  YouTube's own audio. Stretching a chunk takes well under a second; until it's done (right
  after changing the speed) that chunk plays resampled, i.e. briefly higher or lower.
* The extension only hooks the player on `youtube.com` (videos and Shorts), not embeds on other
  sites.

## Development

`inject.js` runs in the page's main world at `document_start`, so it can patch
`MediaSource`/`SourceBuffer` before YouTube's player loads. `bridge.js` (isolated world) relays
messages to `background.js`, which makes the localhost request. Doing that from the
service worker avoids YouTube's CSP and the browser's local-network restrictions.

### Building the app

`packaging/build.py` builds the downloadable app: a portable Python
([python-build-standalone](https://github.com/astral-sh/python-build-standalone), pinned) with
every dependency except PyTorch, plus the server and the extension. PyTorch is left out
because the right build depends on the user's GPU; the first start installs it with
`pip --user` into the data folder (`MR_HOME`, `PYTHONUSERBASE`), which also holds the models.

```bash
python packaging/build.py linux     # dist/MusicRemover-x86_64.AppImage + .tar.gz
python packaging/build.py windows   # then: ISCC.exe packaging\windows\musicremover.iss
```

Each target has to be built on its own OS. `.github/workflows/release.yml` builds both, plus
the Windows installer (Inno Setup), and pushing a tag like `v1.0.0` attaches them to a
GitHub release.
