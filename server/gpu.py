"""GPU detection for NVIDIA (CUDA), AMD (ROCm on Linux and Windows) and Apple (MPS).

AMD GPUs are used through PyTorch's ROCm build, which exposes them through the
same torch.cuda API as NVIDIA cards (torch.version.hip is set instead of
torch.version.cuda), so the models run unchanged on the "cuda" device.

If a GPU is physically present but the installed PyTorch can't use it (the
usual cause: a CPU-only wheel), detect() returns a hint explaining what to
install instead of silently running on the CPU.
"""

import glob
import os
import platform
import subprocess

import torch

VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}

INSTALL_HINTS = {
    "nvidia": "NVIDIA GPU found, but this PyTorch has no CUDA support. Install the CUDA build: "
              "https://pytorch.org/get-started/locally/",
    "amd-linux": "AMD GPU found, but this PyTorch has no ROCm support. Install the ROCm build: "
                 "https://pytorch.org/get-started/locally/ (choose ROCm)",
    "amd-windows": "AMD GPU found, but this PyTorch has no ROCm support. Radeon RX 7000/9000 "
                   "(and some Ryzen AI APUs): install AMD's ROCm PyTorch for Windows (Python 3.12), "
                   "see https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/"
                   "installrad/windows/install-pytorch.html",
}


def _prepare_rocm():
    # Without this, MIOpen benchmarks kernels on first use of each layer shape,
    # which can stall the first chunk for minutes. FAST picks kernels heuristically.
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")


def physical_gpus():
    """Best-effort list of (vendor, name) for GPUs in the machine, independent of PyTorch."""
    found = []
    system = platform.system()
    if system == "Linux":
        for path in sorted(glob.glob("/sys/class/drm/card[0-9]*/device/vendor")):
            try:
                vendor = VENDORS.get(open(path).read().strip().lower())
            except OSError:
                continue
            if vendor:
                found.append((vendor, os.path.basename(os.path.dirname(os.path.dirname(path)))))
    elif system == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        for name in filter(None, (l.strip() for l in out.splitlines())):
            low = name.lower()
            vendor = ("nvidia" if "nvidia" in low else
                      "amd" if ("amd" in low or "radeon" in low) else
                      "intel" if "intel" in low else None)
            if vendor:
                found.append((vendor, name))
    return found


def detect():
    """Returns dict(device, name, backend, hint). device is None when only the CPU is usable."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        if getattr(torch.version, "hip", None):
            _prepare_rocm()
            return dict(device="cuda", name=f"{name} (AMD ROCm)", backend="rocm", hint=None)
        return dict(device="cuda", name=f"{name} (NVIDIA CUDA)", backend="cuda", hint=None)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return dict(device="mps", name="Apple GPU (Metal)", backend="mps", hint=None)

    hint = None
    vendors = {v for v, _ in physical_gpus()}
    if "nvidia" in vendors:
        hint = INSTALL_HINTS["nvidia"]
    elif "amd" in vendors:
        hint = INSTALL_HINTS["amd-windows" if platform.system() == "Windows" else "amd-linux"]
    return dict(device=None, name=None, backend="cpu", hint=hint)
