"""Music removal models. Each engine takes stereo float32 audio [2, T] at
44.1 kHz and returns what should be *kept*, same shape.

  bandit      BandIt Plus (default). Cinematic model trained on DnR: splits
              speech / music / effects; keeps speech + effects.
  demucs      HTDemucs fine-tuned for vocals (MVSep weights, via MSST). About
              6x faster than BandIt, but it splits vocals / everything else,
              so sound effects are removed together with the music.
  dnr_demucs  Hybrid Demucs trained on DnR (the baseline from the BandIt
              paper). Keeps speech + effects like BandIt. Weights come from
              Zenodo (CC-BY-NC 4.0); experimental.
  voc_ft      UVR-MDX-NET-Voc_FT (UVR). Fast; vocals only.
  melband     MelBand RoFormer (Kim, fine-tuned by unwa). Least music bleed of
              all models tested, but the slowest; vocals only.
  voc_ft_dnr  Voices from voc_ft + sound effects from dnr_demucs (experimental).

Shared on top of every model:

  * Bleed suppression. The model's own music estimate drives one extra,
    cheap spectral mask on what we keep:
        keep *= |keep|^2 / (|keep|^2 + strength * |music|^2)
    strength 0 disables it. See bench/README.md for measurements.
  * Speed. On a GPU the model runs in half precision (fp16). If fp16 fails
    or produces non-finite output, it falls back to full precision for the
    rest of the session (MR_FP16=0 disables it).
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
DNR_DEMUCS_URL = "https://zenodo.org/api/records/10160698/files/dnr-demucs.ckpt/content"
UVR_REL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
SEPARATOR_REL = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"


def _fetch(url: str, name: str = None) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name or url.rsplit("/", 1)[-1])
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
    # Register the packages without running their __init__, which imports
    # the whole training stack (asteroid, wandb, ...). Inference doesn't need it.
    for name in ("models", "models.bandit", "models.bandit.core"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(path, *name.split("."))]
            sys.modules[name] = m
    return path


def _state_dict(path: str) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd.get("state_dict", sd) if isinstance(sd, dict) else sd


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
    """Base: a separation network plus which of its stems to keep."""

    keep_stems: tuple = ()
    music_stems: tuple = ()

    def __init__(self):
        self.net = None      # set by subclasses; maps [B, 2, chunk] -> [B, S, 2, chunk]
        self.stems = []
        self.chunk = 0
        self.device = "cpu"
        self.fp16 = False

    def to(self, device: str):
        if device != self.device:
            self.net.to(device)
            self.device = device
        # AMD ROCm reports as "cuda" as well.
        self.fp16 = device == "cuda" and os.environ.get("MR_FP16", "1") != "0"
        return self

    def run(self, b: torch.Tensor) -> torch.Tensor:
        return self.net(b)

    def _forward(self, b: torch.Tensor) -> torch.Tensor:
        if self.fp16:
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    y = self.run(b)
                y = y.float()
                if torch.isfinite(y).all():
                    return y
                reason = "non-finite output"
            except RuntimeError as e:
                reason = str(e).splitlines()[0][:120]
            print(f"[musicremover] fp16 unavailable ({reason}); using full precision", flush=True)
            self.fp16 = False
        return self.run(b)

    @torch.no_grad()
    def stems_of(self, x) -> dict:
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        x = x.to(self.device)
        # 2x window overlap: every sample is processed twice and cross-faded.
        overlap = int(os.environ.get("MR_OVERLAP", 2))
        batch = int(os.environ.get("MR_BATCH", 8 if self.device != "cpu" else 4))
        y = overlap_add(self._forward, x, self.chunk, overlap=overlap, batch=batch)
        return {s: y[i] for i, s in enumerate(self.stems)}

    @torch.no_grad()
    def keep(self, pcm: np.ndarray, strength: float = DEFAULT_STRENGTH) -> np.ndarray:
        stems = self.stems_of(pcm)
        keep = sum(stems[s] for s in self.keep_stems)
        music = sum(stems[s] for s in self.music_stems)
        out = suppress_bleed(keep, music, strength)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return out.cpu().numpy()


class BanditEngine(Engine):
    keep_stems = ("speech", "effects")
    music_stems = ("music",)

    def __init__(self):
        super().__init__()
        _msst()
        cfg_path = _fetch(f"{MSST_REL}/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml")
        with open(cfg_path) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        self.net = MultiMaskMultiSourceBandSplitRNNSimple(**cfg["model"])
        self.net.load_state_dict(_state_dict(_fetch(f"{MSST_REL}/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt")))
        self.net.eval()
        self.stems = list(cfg["model"]["stems"])  # speech, music, effects
        self.chunk = int(cfg["audio"]["chunk_size"])


class DemucsEngine(Engine):
    keep_stems = ("vocals",)
    music_stems = ("other",)

    def __init__(self):
        super().__init__()
        msst = _msst()
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(os.path.join(msst, "configs", "config_vocals_htdemucs.yaml"))
        from models.demucs4ht import get_model
        self.net = get_model(cfg)
        self.net.load_state_dict(_state_dict(_fetch(f"{MSST_REL}/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt")))
        self.net.eval()
        self.stems = list(cfg.training.instruments)  # vocals, other
        self.chunk = int(cfg.audio.chunk_size)       # 11 s, the length it was trained on


class DnRDemucsEngine(Engine):
    keep_stems = ("speech", "effects")
    music_stems = ("music",)

    def __init__(self):
        super().__init__()
        import torchaudio
        self.stems = ["speech", "music", "effects"]
        # Same settings as configs/model/demucs.yaml in kwatcharasupat/bandit (mono model).
        self.net = torchaudio.models.HDemucs(sources=self.stems, audio_channels=1, nfft=4096, depth=6)
        try:
            sd = _state_dict(_fetch(DNR_DEMUCS_URL, "dnr-demucs.ckpt"))
        except OSError as e:
            raise RuntimeError(
                f"Couldn't download the DnR Demucs weights ({e}). Download dnr-demucs.ckpt from "
                f"https://zenodo.org/records/10160698 into {CACHE}") from e
        # Lightning checkpoints prefix keys (e.g. "model.demucs."); keep what follows "demucs.".
        sd = {k.split("demucs.", 1)[1] if "demucs." in k else k: v for k, v in sd.items()}
        self.net.load_state_dict(sd)
        self.net.eval()
        self.chunk = 6 * SR

    def run(self, b: torch.Tensor) -> torch.Tensor:
        # Mono model: fold the stereo channels into the batch and back.
        B, C, T = b.shape
        y = self.net(b.reshape(B * C, 1, T))          # [B*C, S, 1, T]
        return y.reshape(B, C, len(self.stems), T).transpose(1, 2)


class _VocalsOnly(Engine):
    """Single-target vocal models: 'other' is the mix minus the vocals."""

    keep_stems = ("vocals",)
    music_stems = ("other",)

    def vocals(self, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def run(self, b: torch.Tensor) -> torch.Tensor:
        v = self.vocals(b)
        return torch.stack([v, b - v], dim=1)  # [B, 2 stems, C, T]


class VocFTEngine(_VocalsOnly):
    """UVR-MDX-NET-Voc_FT (UVR). The ONNX model is converted to PyTorch with
    onnx2torch so it runs on any GPU PyTorch supports, including AMD ROCm."""

    # From UVR's mdx_model_data.json for this model.
    N_FFT, HOP, DIM_F, DIM_T, COMPENSATE = 7680, 1024, 3072, 256, 1.021

    def __init__(self):
        super().__init__()
        import onnx
        from onnx2torch import convert
        self.net = convert(onnx.load(_fetch(f"{UVR_REL}/UVR-MDX-NET-Voc_FT.onnx"))).eval()
        self.stems = ["vocals", "other"]
        self.chunk = self.HOP * (self.DIM_T - 1)  # 261120 samples -> exactly 256 STFT frames

    def vocals(self, b: torch.Tensor) -> torch.Tensor:
        B, C, T = b.shape
        win = torch.hann_window(self.N_FFT, periodic=True, device=b.device)
        spec = torch.stft(b.reshape(B * C, T).float(), self.N_FFT, self.HOP, window=win,
                          center=True, return_complex=True)                   # [B*C, F, 256]
        x = torch.view_as_real(spec).permute(0, 3, 1, 2)                     # [B*C, 2, F, 256]
        x = x.reshape(B, C * 2, -1, x.shape[-1])[:, :, :self.DIM_F].clone()   # [B, 4, 3072, 256]
        x[:, :, :3] = 0  # UVR zeroes the lowest bins (< ~17 Hz) before the model
        y = self.net(x).float()
        y = torch.nn.functional.pad(y, (0, 0, 0, self.N_FFT // 2 + 1 - self.DIM_F))
        y = y.reshape(B * C, 2, -1, y.shape[-1]).permute(0, 2, 3, 1).contiguous()
        v = torch.istft(torch.view_as_complex(y), self.N_FFT, self.HOP, window=win,
                        center=True, length=T)
        return v.reshape(B, C, T) * self.COMPENSATE


class MelBandEngine(_VocalsOnly):
    """MelBand RoFormer, Kim's vocal model fine-tuned by unwa. Least bleed of the
    models tested, but about 5x slower than Voc_FT."""

    def __init__(self):
        super().__init__()
        _msst()
        with open(_fetch(f"{SEPARATOR_REL}/config_mel_band_roformer_kim_ft_unwa.yaml")) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        from models.bs_roformer.mel_band_roformer import MelBandRoformer
        self.net = MelBandRoformer(**cfg["model"])
        self.net.load_state_dict(_state_dict(_fetch(f"{SEPARATOR_REL}/mel_band_roformer_kim_ft_unwa.ckpt")))
        self.net.eval()
        self.stems = ["vocals", "other"]
        self.chunk = int(cfg["audio"]["chunk_size"])

    def vocals(self, b: torch.Tensor) -> torch.Tensor:
        v = self.net(b)
        return v if v.dim() == 3 else v[:, 0]  # [B, C, T]


class VocFTDnREngine(Engine):
    """Voices from UVR Voc FT + sound effects from DnR Demucs (experimental).

    Voc FT has no effects stem (its "other" is music and effects together), and DnR
    Demucs separates effects from music. Combining them keeps Voc FT's voices and adds
    DnR's effects back. The bleed filter uses DnR's music estimate. Both models run on
    every chunk, and they're shared with the standalone voc_ft / dnr_demucs options, so
    switching between them doesn't load anything twice.
    """

    keep_stems = ("vocals", "effects")
    music_stems = ("music",)

    def __init__(self):
        super().__init__()
        self.voc = get("voc_ft")
        self.dnr = get("dnr_demucs")

    def to(self, device: str):
        self.voc.to(device)
        self.dnr.to(device)
        self.device = device
        return self

    @property
    def fp16(self):
        return self.voc.fp16 and self.dnr.fp16

    @fp16.setter
    def fp16(self, value):
        pass  # set on the two models

    @torch.no_grad()
    def stems_of(self, x) -> dict:
        v = self.voc.stems_of(x)
        d = self.dnr.stems_of(x)
        return {"vocals": v["vocals"], "effects": d["effects"], "music": d["music"]}


ENGINES = {"bandit": BanditEngine, "demucs": DemucsEngine, "dnr_demucs": DnRDemucsEngine,
           "voc_ft": VocFTEngine, "melband": MelBandEngine, "voc_ft_dnr": VocFTDnREngine}

_loaded: dict = {}


def get(name: str) -> Engine:
    """Load an engine once and reuse it (combined engines share their parts)."""
    if name not in _loaded:
        print(f"[musicremover] loading {name}", flush=True)
        _loaded[name] = ENGINES[name]()
    return _loaded[name]
