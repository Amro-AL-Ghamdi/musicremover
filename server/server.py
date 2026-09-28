"""Local music-removal server for the YouTube Music Remover extension.

The browser extension sends ~1 minute chunks of the YouTube audio stream
(exactly the bytes the player appended to its MediaSource buffer: a WebM/Opus
or fMP4/AAC init segment followed by media segments). This server decodes the
chunk with ffmpeg, runs Demucs source separation and returns only the
"vocals" stem (speech + singing, no instruments) encoded as Ogg/Opus.

Run:  python server.py            (listens on http://127.0.0.1:8765)
Env:  MR_MODEL   demucs model name (default: htdemucs; htdemucs_ft = better/slower)
      MR_DEVICE  cuda | mps | cpu (default: auto)
      MR_PORT    port (default: 8765)
"""

import os
import subprocess
import tempfile
import threading
import time

import anyio
import numpy as np
import torch
import uvicorn
from demucs.apply import apply_model
from demucs.pretrained import get_model
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

MODEL_NAME = os.environ.get("MR_MODEL", "htdemucs")
PORT = int(os.environ.get("MR_PORT", "8765"))


def pick_device() -> str:
    forced = os.environ.get("MR_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
print(f"[musicremover] loading {MODEL_NAME} on {DEVICE} ...", flush=True)
MODEL = get_model(MODEL_NAME)
MODEL.to(DEVICE)
MODEL.eval()
SR = MODEL.samplerate
CHANNELS = MODEL.audio_channels
VOCALS = MODEL.sources.index("vocals")
print(f"[musicremover] ready (sources={MODEL.sources}, sr={SR})", flush=True)

# One separation at a time: the model (and GPU) is shared.
LOCK = threading.Lock()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def decode(data: bytes, mime: str) -> np.ndarray:
    """Decode a container chunk to float32 [channels, samples] at the model rate."""
    suffix = ".mp4" if "mp4" in mime else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
             "-vn", "-f", "f32le", "-ac", str(CHANNELS), "-ar", str(SR), "pipe:1"],
            capture_output=True,
        )
    finally:
        os.unlink(path)
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    if pcm.size == 0:
        raise HTTPException(400, f"ffmpeg could not decode chunk: {proc.stderr.decode()[-500:]}")
    return pcm.reshape(-1, CHANNELS).T.copy()


def encode_opus(pcm: np.ndarray) -> bytes:
    interleaved = np.ascontiguousarray(pcm.T, dtype=np.float32).tobytes()
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", str(SR), "-ac", str(CHANNELS), "-i", "pipe:0",
         "-c:a", "libopus", "-b:a", "128k", "-f", "ogg", "pipe:1"],
        input=interleaved, capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(500, f"ffmpeg encode failed: {proc.stderr.decode()[-500:]}")
    return proc.stdout


def separate_vocals(pcm: np.ndarray) -> np.ndarray:
    wav = torch.from_numpy(pcm)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std
    with torch.no_grad():
        sources = apply_model(MODEL, wav[None], device=DEVICE, shifts=1,
                              split=True, overlap=0.25, progress=False)[0]
    vocals = sources[VOCALS] * std + mean
    return vocals.cpu().numpy()


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE}


@app.post("/separate")
async def separate(request: Request, mime: str = "audio/webm"):
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty body")
    # Run the heavy work in a worker thread so the event loop stays free.
    return await anyio.to_thread.run_sync(_separate_sync, data, mime)


def _separate_sync(data: bytes, mime: str) -> Response:
    t0 = time.time()
    pcm = decode(data, mime)
    with LOCK:
        vocals = separate_vocals(pcm)
    out = encode_opus(vocals)
    secs = pcm.shape[1] / SR
    print(f"[musicremover] {secs:.1f}s of audio in {time.time() - t0:.1f}s", flush=True)
    return Response(content=out, media_type="audio/ogg",
                    headers={"X-Duration": f"{secs:.3f}"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
