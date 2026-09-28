"""Separation engines. Each takes stereo float32 audio [2, T] at 44.1 kHz and
returns what should be *kept* (everything except the music), same shape.

  bandit   BandIt Plus (DnR): splits into speech / music / effects and keeps
           speech + effects. Sound effects survive; singing mostly goes with music.
  hybrid   BandIt Plus, then a vocal model on BandIt's music stem to rescue
           singing: keep = speech + effects + vocals(music).
  vocals   A vocal-separation model only (BS-RoFormer). Least music bleed on
           speech, but sound effects are removed together with the music.

See bench/README.md for how they compare.
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

# Model code comes from ZFTurbo/Music-Source-Separation-Training (MIT), pinned.
MSST_REPO = "https://github.com/ZFTurbo/Music-Source-Separation-Training.git"
MSST_COMMIT = "84b1eac0887756b4f1a9d7a1ff49105939749ed2"
MSST_REL = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download"
UVR_REL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"

WEIGHTS = {
    "bandit": (f"{MSST_REL}/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml",
               f"{MSST_REL}/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt"),
    "bs_roformer": ("configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",  # inside MSST repo
                    f"{UVR_REL}/model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
}


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


def _load_state(path: str):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("state_dict", sd) if isinstance(sd, dict) else sd


def overlap_add(fn, x: torch.Tensor, chunk: int, overlap: int = 4, batch: int = 4) -> torch.Tensor:
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


class Model:
    """A MSST model with its config, movable between devices."""

    def __init__(self, kind: str):
        msst = _msst()
        cfg_src, ckpt_url = WEIGHTS[kind]
        cfg_path = os.path.join(msst, cfg_src) if not cfg_src.startswith("http") else _fetch(cfg_src)
        with open(cfg_path) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        if kind == "bandit":
            from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
            self.net = MultiMaskMultiSourceBandSplitRNNSimple(**self.cfg["model"])
            self.stems = list(self.cfg["model"]["stems"])
        else:
            from models.bs_roformer import BSRoformer
            self.net = BSRoformer(**self.cfg["model"])
            self.stems = [self.cfg["training"]["target_instrument"]]
        self.net.load_state_dict(_load_state(_fetch(ckpt_url)))
        self.net.eval()
        self.chunk = int(self.cfg["audio"]["chunk_size"])
        self.device = "cpu"

    def to(self, device: str):
        if device != self.device:
            self.net.to(device)
            self.device = device
        return self

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> dict:
        x = x.to(self.device)

        def fn(b):
            y = self.net(b)
            return y if y.dim() == 4 else y.unsqueeze(1)  # single-target models -> [B, 1, C, T]

        # 4x window overlap is the quality default; on CPU use 2x (half the work).
        overlap = int(os.environ.get("MR_OVERLAP", 4 if self.device != "cpu" else 2))
        y = overlap_add(fn, x, self.chunk, overlap=overlap)
        return {s: y[i] for i, s in enumerate(self.stems)}


class Engine:
    NAMES = ("bandit", "hybrid", "vocals")

    def __init__(self, name: str):
        assert name in self.NAMES, name
        self.name = name
        self.bandit = Model("bandit") if name in ("bandit", "hybrid") else None
        self.vocal = Model("bs_roformer") if name in ("hybrid", "vocals") else None
        self.device = "cpu"

    def to(self, device: str):
        for m in (self.bandit, self.vocal):
            if m:
                m.to(device)
        self.device = device
        return self

    def keep(self, pcm: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32))
        if self.name == "vocals":
            out = self.vocal(x)["vocals"]
        else:
            stems = self.bandit(x)
            out = stems["speech"] + stems["effects"]
            if self.name == "hybrid":
                out = out + self.vocal(stems["music"])["vocals"]
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return out.cpu().numpy()
