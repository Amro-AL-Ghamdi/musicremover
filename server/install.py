"""Install the server's dependencies with the right PyTorch build for your GPU.

    python install.py            detect the GPU and install everything
    python install.py --dry-run  only show what would be installed
    python install.py --target cpu|nvidia|amd|apple   skip detection

PyTorch ships separate builds per GPU vendor under the same package name, so
a plain `pip install torch` gives you the wrong one on many machines (e.g. a
CPU-only build on Windows, or a 3 GB CUDA build on a machine without NVIDIA).
This picks the build, installs it, then installs requirements.txt.

  NVIDIA          CUDA 13.0 build (driver >= 580), or CUDA 12.6 (driver >= 525)
  AMD, Linux      ROCm 7.2 build
  AMD, Windows    AMD's ROCm 7.2.1 wheels (RX 7000/9000, Ryzen AI; needs Python 3.12)
  Apple Silicon   the default build (Metal is included)
  otherwise       the CPU-only build (small download)
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpu  # noqa: E402  (no torch import at module level)

PYTORCH = "https://download.pytorch.org/whl"
AMD_WINDOWS = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
AMD_WINDOWS_TORCH = "2.9.1+rocm7.2.1"
AMD_WINDOWS_GUIDE = ("https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/"
                     "installrad/windows/install-pytorch.html")


def nvidia_driver():
    """Driver version as a tuple, e.g. (580, 65), or None if nvidia-smi isn't available."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def detect_target():
    system, machine = platform.system(), platform.machine().lower()
    if system == "Darwin":
        return "apple" if machine in ("arm64", "aarch64") else "cpu", "macOS"
    vendors = [v for v, _ in gpu.physical_gpus()]
    if "nvidia" in vendors or nvidia_driver():
        return "nvidia", "NVIDIA GPU found"
    if "amd" in vendors:
        return "amd", "AMD GPU found"
    return "cpu", "no NVIDIA/AMD GPU found"


def plan(target):
    """Returns (pip args for torch+torchaudio, expected build kind, notes). Exits on dead ends."""
    system = platform.system()
    notes = []
    if target == "nvidia":
        drv = nvidia_driver()
        if drv and drv < (525, 0):
            notes.append(f"NVIDIA driver {drv[0]}.{drv[1]} is too old for current PyTorch "
                         "(needs 525+); installing the CPU build. Update the driver to use the GPU.")
            return plan("cpu")[0], "cpu", notes
        if drv and drv < (580, 0):
            notes.append(f"NVIDIA driver {drv[0]}.{drv[1]}: using the CUDA 12.6 build "
                         "(driver 580+ gets CUDA 13.0).")
            return ["torch", "torchaudio", "--index-url", f"{PYTORCH}/cu126"], "cuda", notes
        if not drv:
            notes.append("Couldn't read the NVIDIA driver version; assuming a recent driver (580+).")
        return ["torch", "torchaudio", "--index-url", f"{PYTORCH}/cu130"], "cuda", notes
    if target == "amd":
        if system == "Linux":
            return ["torch", "torchaudio", "--index-url", f"{PYTORCH}/rocm7.2"], "rocm", notes
        if system == "Windows":
            if sys.version_info[:2] != (3, 12):
                sys.exit(f"AMD's ROCm PyTorch for Windows needs Python 3.12 (this is "
                         f"{platform.python_version()}). Re-run install.py with Python 3.12, "
                         f"or use --target cpu.")
            notes.append("AMD on Windows supports Radeon RX 7000/9000 and Ryzen AI 300/Max. "
                         "It needs a recent Adrenalin driver. Guide: " + AMD_WINDOWS_GUIDE)
            return [f"torch=={AMD_WINDOWS_TORCH}", f"torchaudio=={AMD_WINDOWS_TORCH}",
                    "--find-links", AMD_WINDOWS], "rocm", notes
        sys.exit("AMD GPUs are supported on Linux and Windows only.")
    if target == "apple":
        return ["torch", "torchaudio"], "mps", notes
    # CPU: the dedicated index avoids pulling ~3 GB of CUDA libraries on Linux.
    if system == "Darwin":
        return ["torch", "torchaudio"], "cpu", notes
    return ["torch", "torchaudio", "--index-url", f"{PYTORCH}/cpu"], "cpu", notes


def installed_torch_kind():
    """'cuda' | 'rocm' | 'cpu' for the torch currently installed (CPU-only macOS builds
    report 'cpu'), or None if torch isn't installed."""
    code = ("import torch;"
            "print('rocm' if getattr(torch.version,'hip',None) else "
            "'cuda' if torch.version.cuda else 'cpu')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def pip(*args, dry):
    cmd = [sys.executable, "-m", "pip", *args]
    print("  $ " + " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["auto", "nvidia", "amd", "apple", "cpu"], default="auto")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    if args.target == "auto":
        target, why = detect_target()
        print(f"Detected: {why} -> {target} build")
    else:
        target = args.target
        print(f"Using --target {target}")
    torch_args, kind, notes = plan(target)
    for n in notes:
        print("Note: " + n)

    current = installed_torch_kind()
    # Apple builds report "cpu" too; the default macOS wheel is the right one either way.
    wrong_kind = current is not None and current != kind and kind != "mps"
    print(f"\n1. PyTorch ({kind})" + (f": replacing the installed {current} build" if wrong_kind else ""))
    if wrong_kind:
        # Same package name across builds, so pip would otherwise keep the wrong one.
        pip("uninstall", "-y", "torch", "torchaudio", dry=args.dry_run)
    pip("install", *torch_args, dry=args.dry_run)

    print("\n2. Other dependencies")
    pip("install", "-r", os.path.join(HERE, "requirements.txt"), dry=args.dry_run)

    if args.dry_run:
        return
    print("\n3. Check")
    r = subprocess.run([sys.executable, "-c",
                        "import gpu; d = gpu.detect(); print(d['name'] or 'CPU only'); "
                        "print(d['hint'] or '')"], cwd=HERE, capture_output=True, text=True)
    lines = (r.stdout or r.stderr).strip().splitlines()
    print(f"  PyTorch sees: {lines[0] if lines else 'error'}")
    for extra in lines[1:]:
        if extra:
            print("  " + extra)
    print("\nDone. Start the server with:  python server.py")


if __name__ == "__main__":
    main()
