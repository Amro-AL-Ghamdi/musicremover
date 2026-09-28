"""Music removal with BandIt Plus.

BandIt Plus is a cinematic source-separation model (trained on the DnR
dataset): it splits audio into speech / music / effects. We keep
speech + effects, so sound effects survive and instruments go.

Two things on top of the plain model:

  * Bleed suppression. BandIt's own music estimate is used for one extra,
    cheap spectral mask on what we keep:
        keep *= |keep|^2 / (|keep|^2 + strength * |music|^2)
    Where the model thinks there's still music, the kept audio is pushed down
    further. strength 0 disables it; higher removes more music bleed at a small
    cost to effects that overlap music. See bench/README.md for measurements.
  * Speed. On a GPU the model runs in half precision (fp16), with a larger
    batch. If fp16 fails or produces non-finite output, it falls back to
    full precision for the rest of the session (MR_FP16=0 disables it).
"""

import os
import subprocess
import sys
import types
import urllib.request

import numpy as np
import torch
import yaml

SR = 44100
CACHE = os.environ.get("MR_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"))
DEFAULT_STRENGTH = 1.0

# Model code comes from ZFTurbo/Music-Source-Separation-Training (MIT), pinned.
MSST_REPO = "https://github.com/ZFTurbo/Music-Source-Separation-Training.git"
MSST_COMMIT = "84b1eac0887756b4f1a9d7a1ff49105939749ed2"
MSST_REL = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download"
BANDIT_CONFIG = f"{MSST_REL}/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml"
BANDIT_WEIGHTS = f"{MSST_REL}/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt"


def _fetch(url: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        print(f"[musicremover] downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def _msst() -> str:
    path = os.path.join(CACHE, "msst")
    if not os.path.isdir(os.path.join(path, "models")):
        print("[musicremover] fetching model code (MSST)", flush=True)
        os.makedirs(path, exist_ok=True)
        run = lambda *a: subprocess.run(["git", "-C", path, *a], check=True, capture_output=True)
        run("init", "-q")
        run("fetch", "-q", "--depth", "1", MSST_REPO, MSST_COMMIT)
        run("checkout", "-q", "FETCH_HEAD")
    if path not in sys.path:
        sys.path.insert(0, path)
    # Register the bandit package without running its __init__, which imports
    # the whole training stack (asteroid, wandb, ...). Inference doesn't need it.
    for name in ("models", "models.bandit", "models.bandit.core"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(path, *name.split("."))]
            sys.modules[name] = m
    return path


def overlap_add(fn, x: torch.Tensor, chunk: int, overlap: int, batch: int) -> torch.Tensor:
    """Run fn on overlapping windows of x [C, T] and cross-fade the results.
    fn maps [B, C, chunk] -> [B, S, C, chunk]. Returns [S, C, T]."""
    step = chunk // overlap
    T = x.shape[-1]
    pad = chunk - step
    xp = torch.nn.functional.pad(x, (pad, pad + chunk))
    starts = list(range(0, T + pad, step))
    win = torch.hann_window(chunk, periodic=True, device=x.device) + 1e-3
    out = norm = None
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        y = fn(torch.stack([xp[:, a:a + chunk] for a in s]))
        if out is None:
            out = torch.zeros(y.shape[1], x.shape[0], xp.shape[-1], device=x.device)
            norm = torch.zeros(xp.shape[-1], device=x.device)
        for j, a in enumerate(s):
            out[..., a:a + chunk] += y[j] * win
            norm[a:a + chunk] += win
    return (out / norm.clamp_min(1e-6))[..., pad:pad + T]


def suppress_bleed(keep: torch.Tensor, music: torch.Tensor, strength: float) -> torch.Tensor:
    """Extra spectral mask on `keep` [C, T] wherever the model's music estimate is strong."""
    if strength <= 0:
        return keep
    n_fft, hop = 2048, 512
    win = torch.hann_window(n_fft, device=keep.device)
    K = torch.stft(keep, n_fft, hop, window=win, return_complex=True)
    M = torch.stft(music, n_fft, hop, window=win, return_complex=True)
    pk, pm = K.abs() ** 2, M.abs() ** 2
    mask = pk / (pk + strength * pm + 1e-10)
    return torch.istft(K * mask, n_fft, hop, window=win, length=keep.shape[-1])


class Engine:
    """BandIt Plus, movable between devices. keep() returns speech + effects."""

    def __init__(self):
        _msst()
        with open(_fetch(BANDIT_CONFIG)) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        self.net = MultiMaskMultiSourceBandSplitRNNSimple(**self.cfg["model"])
        sd = torch.load(_fetch(BANDIT_WEIGHTS), map_location="cpu", weights_only=False)
        self.net.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd)
        self.net.eval()
        self.stems = list(self.cfg["model"]["stems"])  # speech, music, effects
        self.chunk = int(self.cfg["audio"]["chunk_size"])
        self.device = "cpu"
        self.fp16 = False

    def to(self, device: str):
        if device != self.device:
            self.net.to(device)
            self.device = device
        # AMD ROCm reports as "cuda" as well.
        self.fp16 = device == "cuda" and os.environ.get("MR_FP16", "1") != "0"
        return self

    def _forward(self, b: torch.Tensor) -> torch.Tensor:
        if self.fp16:
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    y = self.net(b)
                y = y.float()
                if torch.isfinite(y).all():
                    return y
                reason = "non-finite output"
            except RuntimeError as e:
                reason = str(e).splitlines()[0][:120]
            print(f"[musicremover] fp16 unavailable ({reason}); using full precision", flush=True)
            self.fp16 = False
        return self.net(b)

    @torch.no_grad()
    def stems_of(self, pcm: np.ndarray) -> dict:
        x = torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32)).to(self.device)
        # 2x window overlap: every sample is processed twice and cross-faded.
        overlap = int(os.environ.get("MR_OVERLAP", 2))
        batch = int(os.environ.get("MR_BATCH", 8 if self.device != "cpu" else 4))
        y = overlap_add(self._forward, x, self.chunk, overlap=overlap, batch=batch)
        return {s: y[i] for i, s in enumerate(self.stems)}

    @torch.no_grad()
    def keep(self, pcm: np.ndarray, strength: float = DEFAULT_STRENGTH) -> np.ndarray:
        stems = self.stems_of(pcm)
        out = suppress_bleed(stems["speech"] + stems["effects"], stems["music"], strength)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return out.cpu().numpy()
