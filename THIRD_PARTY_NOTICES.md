# Third-party notices

The code in this repository is MIT-licensed (see [LICENSE](LICENSE)). It does not contain
any model weights: those are downloaded on first use, from their original publishers, and
stay under their own licenses. Check them before any commercial use.

## Models (downloaded at run time)

| Model | Source | License |
|---|---|---|
| DnR Demucs (Hybrid Demucs trained on Divide and Remaster), used by `dnr_demucs` and the default `voc_ft_dnr` | [Zenodo record 10160698](https://zenodo.org/records/10160698), from the paper *A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation* (Watcharasupat et al.) | **CC BY-NC 4.0: non-commercial use only** |
| UVR-MDX-NET-Voc_FT, used by `voc_ft`, `voc_ft_int8` and `voc_ft_dnr` | [Ultimate Vocal Remover model repository](https://github.com/TRvlvr/model_repo) | as published by the [UVR project](https://github.com/Anjok07/ultimatevocalremovergui) (MIT); see their repositories |

Because the default model uses the DnR Demucs weights, **using this app with its default
settings is limited to non-commercial use**. `voc_ft` / `voc_ft_int8` don't use them.

## Software used (installed by `install.py` / bundled in the app)

| Package | License |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| torchaudio (HDemucs model code) | BSD-2-Clause |
| ONNX, onnx2torch | Apache-2.0 |
| ONNX Runtime | MIT |
| NumPy | BSD-3-Clause |
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| imageio-ffmpeg | BSD-2-Clause; the FFmpeg binary it ships is LGPL/GPL (see [FFmpeg's license](https://ffmpeg.org/legal.html)) |
| AMD ROCm PyTorch wheels (AMD GPUs) | see AMD's and PyTorch's terms |
| python-build-standalone (Python in the downloadable app) | CPython: PSF License; see [its licenses](https://github.com/astral-sh/python-build-standalone) |

## Benchmark data (`bench/`, downloaded only when you run it)

| Data | License |
|---|---|
| LibriSpeech excerpts via [librosa/data](https://github.com/librosa/data) | CC BY 4.0 |
| Music clips via librosa/data | Creative Commons (see each file's credits in librosa/data) |
| [ESC-50](https://github.com/karolpiczak/ESC-50) sound effects | CC BY-NC 3.0 |

The icon in `packaging/` was made for this project and is covered by the MIT license.
