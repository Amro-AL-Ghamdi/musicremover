"""Local music-removal server for the YouTube Music Remover extension.

The browser extension sends ~1 minute chunks of the YouTube audio stream
(exactly the bytes the player appended to its MediaSource buffer: a WebM/Opus
or fMP4/AAC init segment followed by media segments). This server decodes the
chunk with ffmpeg, removes the music (see engines.py) and returns the rest
(speech, sound effects and, depending on the engine, singing) as Ogg/Opus.

Run:  python server.py            (listens on http://127.0.0.1:8765)
Env:  MR_ENGINE  default engine: bandit | hybrid | vocals (default: bandit)
      MR_DEVICE  force a device: cuda | mps | cpu (default: GPU if available).
                 AMD GPUs (ROCm build of PyTorch) are "cuda" too.
      MR_PORT    port (default: 8765)

The extension can override engine and device per request (?engine=&device=),
which is how the popup's "Force CPU" switch works.
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
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

import engines
import gpu

SR = engines.SR
CHANNELS = 2
PORT = int(os.environ.get("MR_PORT", "8765"))
DEFAULT_ENGINE = os.environ.get("MR_ENGINE", "bandit")


GPU_INFO = gpu.detect()
GPU, GPU_NAME = GPU_INFO["device"], GPU_INFO["name"]


def resolve_device(requested: str) -> str:
    if os.environ.get("MR_DEVICE"):
        return os.environ["MR_DEVICE"]
    if requested == "cpu" or GPU is None:
        return "cpu"
    return GPU


print(f"[musicremover] GPU: {GPU_NAME or 'none usable, using CPU'}", flush=True)
if GPU_INFO["hint"]:
    print(f"[musicremover] {GPU_INFO['hint']}", flush=True)

# One separation at a time: models (and the GPU) are shared.
LOCK = threading.Lock()
ENGINES: dict = {}
LAST = {"engine": None, "device": None}


def get_engine(name: str, device: str) -> "engines.Engine":
    if name not in ENGINES:
        print(f"[musicremover] loading engine '{name}'", flush=True)
        ENGINES[name] = engines.Engine(name)
    eng = ENGINES[name]
    if eng.device != device:
        print(f"[musicremover] engine '{name}' -> {device}", flush=True)
        eng.to(device)
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return eng


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def decode(data: bytes, mime: str) -> np.ndarray:
    """Decode a container chunk to float32 [channels, samples] at 44.1 kHz."""
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


@app.get("/health")
def health():
    return {"ok": True, "gpu": GPU_NAME, "gpu_device": GPU, "gpu_backend": GPU_INFO["backend"],
            "gpu_hint": GPU_INFO["hint"], "engines": list(engines.Engine.NAMES),
            "default_engine": DEFAULT_ENGINE, "last_engine": LAST["engine"], "last_device": LAST["device"],
            "forced_device": os.environ.get("MR_DEVICE")}


@app.post("/separate")
async def separate(request: Request, mime: str = "audio/webm", engine: str = "", device: str = "auto"):
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty body")
    engine = engine or DEFAULT_ENGINE
    if engine not in engines.Engine.NAMES:
        raise HTTPException(400, f"unknown engine {engine!r}")
    # Run the heavy work in a worker thread so the event loop stays free.
    return await anyio.to_thread.run_sync(_separate_sync, data, mime, engine, resolve_device(device))


def _separate_sync(data: bytes, mime: str, engine: str, device: str) -> Response:
    t0 = time.time()
    pcm = decode(data, mime)
    with LOCK:
        kept = get_engine(engine, device).keep(pcm)
        LAST.update(engine=engine, device=device)
    out = encode_opus(kept)
    secs = pcm.shape[1] / SR
    took = time.time() - t0
    print(f"[musicremover] {secs:.1f}s of audio in {took:.1f}s ({engine} on {device})", flush=True)
    return Response(content=out, media_type="audio/ogg",
                    headers={"X-Duration": f"{secs:.3f}", "X-Device": device, "X-Engine": engine})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
