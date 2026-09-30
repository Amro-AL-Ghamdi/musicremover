"""Build the downloadable app: a portable Python with every dependency except PyTorch
(which depends on the GPU), the server code and the extension. On the first start the
app downloads PyTorch for the user's GPU and the models (see server/launcher.py).

    python packaging/build.py linux     -> dist/MusicRemover-x86_64.AppImage
                                           dist/MusicRemover-linux-x86_64.tar.gz
    python packaging/build.py windows   -> build/windows/MusicRemover, which
                                           packaging/windows/musicremover.iss turns into
                                           dist/MusicRemover-Setup.exe

Run it on the OS you build for (pip installs that OS's packages). The GitHub Actions
workflow (.github/workflows/release.yml) does both and attaches the files to a release.
"""

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "packaging")
DIST = os.path.join(ROOT, "dist")

# Portable CPython builds (astral-sh/python-build-standalone), pinned for reproducible
# builds. Windows needs 3.12: AMD's ROCm PyTorch for Windows only exists for 3.12.
PBS = "https://github.com/astral-sh/python-build-standalone/releases/download/20250902/"
PYTHON = {
    "linux": PBS + "cpython-3.13.7%2B20250902-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
    "windows": PBS + "cpython-3.12.11%2B20250902-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
}
APPIMAGETOOL = ("https://github.com/AppImage/appimagetool/releases/download/continuous/"
                "appimagetool-x86_64.AppImage")

# Installed on the user's machine instead, from the index that matches their GPU.
GPU_PACKAGES = {"torch", "torchaudio", "torchvision"}
# Needs torch at install time only as a declared dependency; bundled without it.
NO_DEPS = {"onnx2torch"}


def log(msg):
    print(f"==> {msg}", flush=True)


def download(url, path):
    if not os.path.exists(path):
        log(f"downloading {url}")
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def requirements():
    """Package names from server/requirements.txt, split into (bundled, no-deps)."""
    names = []
    with open(os.path.join(ROOT, "server", "requirements.txt")) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
    base = lambda n: re.split(r"[<>=!~\s\[;]", n, maxsplit=1)[0].lower()
    bundled = [n for n in names if base(n) not in GPU_PACKAGES | NO_DEPS]
    no_deps = [n for n in names if base(n) in NO_DEPS]
    return bundled, no_deps


def python_exe(app, target):
    return os.path.join(app, "python", "python.exe" if target == "windows" else "bin/python3")


def build_app(target, work, python_url):
    """build/<target>/MusicRemover: python/ + app/ (server, extension)."""
    app = os.path.join(work, "MusicRemover")
    if os.path.isdir(app):
        shutil.rmtree(app)
    os.makedirs(app)

    archive = download(python_url, os.path.join(work, python_url.rsplit("/", 1)[-1].replace("%2B", "+")))
    log("unpacking Python")
    with tarfile.open(archive) as t:  # contains python/
        if hasattr(tarfile, "data_filter"):
            t.extractall(app, filter="data")
        else:
            t.extractall(app)
    py = python_exe(app, target)

    bundled, no_deps = requirements()
    log(f"installing {', '.join(bundled + no_deps)}")
    pip = [py, "-m", "pip", "install", "--no-cache-dir", "--no-warn-script-location",
           "--disable-pip-version-check"]
    subprocess.run(pip + bundled, check=True)
    if no_deps:
        subprocess.run(pip + ["--no-deps"] + no_deps, check=True)

    log("copying the server and the extension")
    server = os.path.join(app, "app", "server")
    os.makedirs(server)
    for name in os.listdir(os.path.join(ROOT, "server")):
        if name.endswith(".py") or name == "requirements.txt":
            shutil.copy2(os.path.join(ROOT, "server", name), server)
    shutil.copytree(os.path.join(ROOT, "extension"), os.path.join(app, "app", "extension"))
    for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(os.path.join(ROOT, name), app)

    # Compile now: the AppImage is read-only, so Python couldn't cache them at run time.
    log("compiling")
    subprocess.run([py, "-m", "compileall", "-q", "-j", "0", app], check=False)
    return app


def build_linux(work, python_url, appimage=True):
    app = build_app("linux", work, python_url)
    for name in ("AppRun", "musicremover.desktop"):
        shutil.copy2(os.path.join(PKG, "linux", name), app)
    shutil.copy2(os.path.join(PKG, "linux", "AppRun"), os.path.join(app, "run.sh"))
    shutil.copy2(os.path.join(ROOT, "stop.sh"), app)
    for name in ("AppRun", "run.sh", "stop.sh"):
        path = os.path.join(app, name)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    shutil.copy2(os.path.join(PKG, "icon.png"), os.path.join(app, "musicremover.png"))
    shutil.copy2(os.path.join(PKG, "icon.png"), os.path.join(app, ".DirIcon"))

    os.makedirs(DIST, exist_ok=True)
    tgz = os.path.join(DIST, "MusicRemover-linux-x86_64.tar.gz")
    log(f"writing {tgz}")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(app, arcname="MusicRemover")

    if appimage:
        tool = download(APPIMAGETOOL, os.path.join(work, "appimagetool-x86_64.AppImage"))
        os.chmod(tool, 0o755)
        out = os.path.join(DIST, "MusicRemover-x86_64.AppImage")
        log(f"writing {out}")
        # CI machines often have no FUSE: let appimagetool unpack itself instead of mounting.
        env = dict(os.environ, ARCH="x86_64", APPIMAGE_EXTRACT_AND_RUN="1")
        subprocess.run([tool, "--no-appstream", app, out], check=True, env=env)


def build_windows(work, python_url):
    app = build_app("windows", work, python_url)
    shutil.copy2(os.path.join(PKG, "windows", "MusicRemover.bat"), app)
    shutil.copy2(os.path.join(ROOT, "stop.bat"), os.path.join(app, "Stop MusicRemover.bat"))
    shutil.copy2(os.path.join(PKG, "windows", "musicremover.ico"), app)
    log("next: ISCC.exe packaging\\windows\\musicremover.iss -> dist\\MusicRemover-Setup.exe")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=["linux", "windows"])
    ap.add_argument("--python-url", help="portable Python archive to use instead of the pinned one")
    ap.add_argument("--no-appimage", action="store_true", help="Linux: only build the .tar.gz")
    args = ap.parse_args()
    work = os.path.join(ROOT, "build", args.target)
    os.makedirs(work, exist_ok=True)
    url = args.python_url or PYTHON[args.target]
    if args.target == "linux":
        build_linux(work, url, appimage=not args.no_appimage)
    else:
        build_windows(work, url)


if __name__ == "__main__":
    sys.exit(main())
