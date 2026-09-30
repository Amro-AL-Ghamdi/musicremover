"""Music removal models. Each engine takes stereo float32 audio [2, T] at
44.1 kHz and returns what should be *kept*, same shape.

  voc_ft_dnr  (default) Voices from voc_ft + sound effects from dnr_demucs.
  voc_ft      UVR-MDX-NET-Voc_FT (UVR). Fast; keeps vocals only, so sound effects
              are removed together with the music.
  voc_ft_int8 Voc FT quantized to int8 for weaker machines: on the CPU it runs with
              onnxruntime, about 2x faster than voc_ft there; on a GPU it runs
              the regular Voc FT.
  dnr_demucs  Hybrid Demucs trained on DnR (the baseline from the BandIt paper):
              splits speech / music / effects; keeps speech + effects. Weights
              come from Zenodo (CC-BY-NC 4.0).

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
import shutil
import urllib.request

import numpy as np
import torch

SR = 44100
CACHE = os.environ.get("MR_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"))
DEFAULT_STRENGTH = 1.0

DNR_DEMUCS_URL = "https://zenodo.org/api/records/10160698/files/dnr-demucs.ckpt/content"
UVR_REL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"


def _fetch(url: str, name: str = None) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name or url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        print(f"[musicremover] downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def ffmpeg_exe() -> str:
    """ffmpeg from PATH, else the copy bundled with the imageio-ffmpeg package, so users
    don't have to install ffmpeg themselves (it includes the libopus encoder we need)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not found: install it, or `pip install imageio-ffmpeg`") from e


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
        y = overlap_add(self._forward, x, self.chunk, overlap=overlap, batch=self.batch())
        return {s: y[i] for i, s in enumerate(self.stems)}

    def parts(self) -> list:
        """The engines whose networks this one runs (itself, or a combination's parts)."""
        return [self]

    batch_override = None  # set by the server after running out of GPU memory

    def batch(self) -> int:
        """Windows per model call."""
        if self.batch_override:
            return self.batch_override
        return int(os.environ.get("MR_BATCH", 8 if self.device != "cpu" else 4))

    def apply(self, stems: dict, strength: float) -> torch.Tensor:
        """What to keep from the stems, with bleed suppression. Shared by the server
        (keep) and the benchmark, so both measure the same thing."""
        keep = sum(stems[s] for s in self.keep_stems)
        music = sum(stems[s] for s in self.music_stems)
        return suppress_bleed(keep, music, strength)

    @torch.no_grad()
    def keep(self, pcm: np.ndarray, strength: float = DEFAULT_STRENGTH) -> np.ndarray:
        out = self.apply(self.stems_of(pcm), strength)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return out.cpu().numpy()


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


class VocFTInt8Engine(VocFTEngine):
    """Voc FT for weaker machines: its frequency layers (MatMul, most of the work besides
    the convolutions) quantized to int8, run by onnxruntime on the CPU.

    It follows the device like every model: on the CPU (no usable GPU, or Force CPU) it
    runs the int8 model; on a GPU it runs the regular Voc FT there in half precision
    (the same network, faster than int8 on a CPU), shared with the voc_ft option.

    Only the MatMuls are quantized, dynamically (activation ranges measured per call):
    full static int8 (convolutions too) was faster still but lost ~5 dB of speech and let
    more music through in pauses, because the model's activations span too wide a range
    for 8 bits. On the benchmark mixes (4-core CPU, Normal bleed suppression) it matches
    voc_ft within measurement noise except speech, 0.4 dB softer, at about half the CPU
    time; see bench/README.md.
    The quantized file is made from the downloaded model on first use (a few seconds).
    """

    def __init__(self):
        self.gpu = None  # the regular Voc FT, while running on a GPU
        Engine.__init__(self)
        import onnxruntime as ort
        path = os.path.join(CACHE, "UVR-MDX-NET-Voc_FT.int8.onnx")
        if not os.path.exists(path):
            from onnxruntime.quantization import QuantType, quantize_dynamic
            src = _fetch(f"{UVR_REL}/UVR-MDX-NET-Voc_FT.onnx")
            print("[musicremover] quantizing Voc FT to int8 (once)", flush=True)
            tmp = path + ".part.onnx"
            quantize_dynamic(src, tmp, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                             per_channel=True)
            os.replace(tmp, path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.net = lambda x: torch.from_numpy(self.session.run(None, {"input": x.cpu().numpy()})[0])
        self.stems = ["vocals", "other"]
        self.chunk = self.HOP * (self.DIM_T - 1)

    # Where it runs follows the regular model while that's in use (the server may move it
    # back to the CPU to make room for another model; then int8 takes over again).
    def _on_gpu(self) -> bool:
        return self.gpu is not None and self.gpu.device != "cpu"

    @property
    def device(self):
        return self.gpu.device if self._on_gpu() else "cpu"

    @device.setter
    def device(self, value):
        pass  # set through to()

    @property
    def fp16(self):
        return self.gpu.fp16 if self._on_gpu() else False

    @fp16.setter
    def fp16(self, value):
        pass

    def to(self, device: str):
        if device == "cpu":
            if self._on_gpu():
                self.gpu.to("cpu")
            self.gpu = None
        else:
            self.gpu = get("voc_ft").to(device)
        return self

    def parts(self) -> list:
        return [self.gpu] if self.gpu is not None else [self]

    def stems_of(self, x) -> dict:
        if self._on_gpu():
            return self.gpu.stems_of(x)
        return super().stems_of(x)

    def batch(self) -> int:
        if self._on_gpu():
            return self.gpu.batch()
        # int8 on the CPU: one window at a time is fastest and needs the least memory
        # (measured: batch 1 0.85 s per audio second / 3.5 GB peak, batch 4 1.11 s / 10.7 GB).
        return self.batch_override or int(os.environ.get("MR_BATCH", 1))


class VocFTDnREngine(Engine):
    """Voices from UVR Voc FT + sound effects from DnR Demucs (the default).

    Voc FT has no effects stem (its "other" is music and effects together), and DnR
    Demucs separates effects from music. Combining them keeps Voc FT's voices and adds
    DnR's effects back. Both models run on every chunk, and they're shared with the
    standalone voc_ft / dnr_demucs options, so switching doesn't load anything twice.

    Each part gets its own bleed filter, driven by the model it came from:
      voices  = filter(Voc FT vocals,  Voc FT other,  VOCAL_FACTOR * strength)
      effects = filter(DnR effects,    DnR music,     strength)
    Measured with a single DnR-driven filter, pauses only reached -48 dB because the
    music Voc FT lets through wasn't caught; Voc FT's own filter handles that (-69 dB
    in pauses on its own at strength 16).
    """

    keep_stems = ("vocals", "effects")
    music_stems = ("music",)
    VOCAL_FACTOR = 4  # Voc FT's filter works best ~4x higher (standalone levels 16/64)

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

    def parts(self) -> list:
        return [self.voc, self.dnr]

    @torch.no_grad()
    def stems_of(self, x) -> dict:
        v = self.voc.stems_of(x)
        d = self.dnr.stems_of(x)
        return {"vocals": v["vocals"], "voc_other": v["other"], "effects": d["effects"], "music": d["music"]}

    def apply(self, stems: dict, strength: float) -> torch.Tensor:
        voices = suppress_bleed(stems["vocals"], stems["voc_other"], self.VOCAL_FACTOR * strength)
        effects = suppress_bleed(stems["effects"], stems["music"], strength)
        return voices + effects


ENGINES = {"voc_ft_dnr": VocFTDnREngine, "voc_ft": VocFTEngine, "voc_ft_int8": VocFTInt8Engine,
           "dnr_demucs": DnRDemucsEngine}
DEFAULT_ENGINE = "voc_ft_dnr"

_loaded: dict = {}


def loaded() -> dict:
    return dict(_loaded)


def get(name: str) -> Engine:
    """Load an engine once and reuse it (combined engines share their parts)."""
    if name not in _loaded:
        print(f"[musicremover] loading {name}", flush=True)
        _loaded[name] = ENGINES[name]()
    return _loaded[name]
