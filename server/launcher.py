"""Start the server, installing first if the dependencies are missing or out of date.

Used by run.sh / run.bat. After a `git pull` that changed requirements.txt, the next
start runs the installer before starting the server, so users don't have to remember to.
"""

import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
# Written by install.py after a successful install; lives inside the venv.
STAMP = os.path.join(sys.prefix, ".musicremover-installed")


def requirements_hash() -> str:
    with open(REQUIREMENTS, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def needs_install() -> bool:
    try:
        with open(STAMP) as f:
            return f.read().strip() != requirements_hash()
    except OSError:
        return True


def main():
    if needs_install():
        print("[musicremover] dependencies missing or changed: running the installer", flush=True)
        r = subprocess.call([sys.executable, os.path.join(HERE, "install.py"), "--no-prefetch"], cwd=HERE)
        if r != 0:
            print("[musicremover] install failed; see the messages above", flush=True)
            sys.exit(r)
    sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "server.py")] + sys.argv[1:], cwd=HERE))


if __name__ == "__main__":
    main()
