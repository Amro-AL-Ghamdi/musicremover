"""Build the downloadable app: a portable Python with every dependency except PyTorch
(which depends on the GPU) and imageio-ffmpeg (GPLv3 FFmpeg binary), the server code and
the extension. On the first start the app downloads those two and the models from their
publishers (see server/launcher.py). The license texts of everything bundled are
collected into licenses/ in the app (see collect_licenses).

    python packaging/build.py linux     -> dist/MusicRemover-x86_64.AppImage
                                           dist/MusicRemover-linux-x86_64.tar.gz
    python packaging/build.py windows   -> build/windows/MusicRemover, which
                                           packaging/windows/musicremover.iss turns into
                                           dist/MusicRemover-Setup.exe

Run it on the OS you build for (pip installs that OS's packages). The GitHub Actions
workflow (.github/workflows/release.yml) does both and attaches the files to a release.
"""

import argparse
import json
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
# Installed on the user's machine at first start (install.py, from PyPI) instead of being
# bundled: its FFmpeg binary is GPLv3, so shipping it inside the app would mean
# distributing the FFmpeg source with every release. A system ffmpeg is used if present.
FIRST_START = {"imageio-ffmpeg"}
# These declare torch as a dependency, so installing them normally would bundle PyTorch
# and its CUDA libraries (several GB, under NVIDIA's license). They're bundled without
# their dependencies; the ones the code needs are in requirements.txt (einops, julius).
NO_DEPS = {"onnx2torch", "demucs", "julius"}
# License texts kept in the repo: the libraries compiled into the bundled Python
# (python-build-standalone ships them only in its "full" archives), and fallback texts
# for packages that don't include their license file.
LICENSES = os.path.join(PKG, "licenses")
FALLBACK_LICENSES = {"flatbuffers": "Apache-2.0.txt"}


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
    bundled = [n for n in names if base(n) not in GPU_PACKAGES | FIRST_START | NO_DEPS]
    no_deps = [n for n in names if base(n) in NO_DEPS]
    return bundled, no_deps


# Run with the bundled Python: every installed distribution with its license files.
_LIST_LICENSES = r"""
import importlib.metadata as m, json, re
pat = re.compile(r"(LICEN[CS]E|COPYING|NOTICE|ThirdPartyNotices)", re.I)
out = []
for d in m.distributions():
    files = [[f.as_posix(), str(d.locate_file(f))] for f in (d.files or []) if pat.search(f.name)]
    out.append({"name": d.metadata["Name"], "version": d.version, "files": files})
print(json.dumps(out))
"""


def collect_licenses(app, py):
    """licenses/ in the app: the license and notice files of every bundled package, and
    of the libraries compiled into the bundled Python. Fails if a package has neither a
    license file nor an entry in FALLBACK_LICENSES, so nothing is shipped unlicensed."""
    dst = os.path.join(app, "licenses")
    shutil.copytree(os.path.join(LICENSES, "python-build-standalone"), os.path.join(dst, "python"))
    out = subprocess.run([py, "-c", _LIST_LICENSES], check=True, capture_output=True, text=True)
    missing = []
    for d in sorted(json.loads(out.stdout), key=lambda d: d["name"].lower()):
        folder = os.path.join(dst, "packages", f"{d['name']}-{d['version']}")
        fallback = FALLBACK_LICENSES.get(d["name"].lower())
        if not d["files"] and not fallback:
            missing.append(d["name"])
            continue
        os.makedirs(folder, exist_ok=True)
        for rel, path in d["files"]:
            # Flatten the path inside site-packages, so same-named files don't collide.
            name = rel.replace("../", "").replace("/", "__")
            shutil.copy2(path, os.path.join(folder, name))
        if fallback:
            shutil.copy2(os.path.join(LICENSES, "fallback", fallback), folder)
    if missing:
        sys.exit(f"no license file found for: {', '.join(missing)} "
                 "(add them to FALLBACK_LICENSES in packaging/build.py)")
    log(f"collected licenses into {dst}")


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
    collect_licenses(app, py)

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
