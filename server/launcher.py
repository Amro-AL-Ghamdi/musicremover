"""Start the server, installing first if the dependencies are missing or out of date.

Used by run.sh / run.bat. After a `git pull` that changed requirements.txt, the next
start runs the installer before starting the server, so users don't have to remember to.

The downloadable app (AppImage / Windows setup, see packaging/) also starts here. Its
own files are read-only, so it sets MR_HOME to a per-user data folder: PyTorch for the
user's GPU (pip --user, PYTHONUSERBASE), the models and the install stamp go there, and
the extension is copied there so Chrome can load it.
"""

import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
# Data folder of the downloadable app; unset when running from a source checkout.
APP_HOME = os.environ.get("MR_HOME")
# Written by install.py after a successful install; lives inside the venv (or APP_HOME).
STAMP = os.path.join(APP_HOME or sys.prefix, ".musicremover-installed")


def requirements_hash() -> str:
    with open(REQUIREMENTS, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def needs_install() -> bool:
    try:
        with open(STAMP) as f:
            return f.read().strip() != requirements_hash()
    except OSError:
        return True


def copy_extension() -> str:
    """App only: copy the extension to the data folder (refreshed on every start, so an
    updated app updates it too) and return where it is."""
    src = os.path.join(os.path.dirname(HERE), "extension")
    dst = os.path.join(APP_HOME, "extension")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def main():
    if APP_HOME:
        os.makedirs(APP_HOME, exist_ok=True)
        os.environ["MR_EXTENSION"] = ext = copy_extension()
        print(f"[musicremover] extension folder (load it once in chrome://extensions with "
              f"Developer mode -> Load unpacked): {ext}", flush=True)
    if needs_install():
        print("[musicremover] dependencies missing or changed: running the installer", flush=True)
        # The app's first start also downloads the default models, so the first video
        # doesn't wait on them. For source checkouts install.sh / install.bat did that.
        first_app_start = APP_HOME and not os.path.exists(STAMP)
        args = [] if first_app_start else ["--no-prefetch"]
        r = subprocess.call([sys.executable, os.path.join(HERE, "install.py"), *args], cwd=HERE)
        if r != 0:
            print("[musicremover] install failed; see the messages above", flush=True)
            sys.exit(r)
    sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "server.py")] + sys.argv[1:], cwd=HERE))


if __name__ == "__main__":
    main()
