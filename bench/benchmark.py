"""Measure BandIt Plus music removal on mixes with known ground truth.

Each test mix = speech + non-music sound effects + background music, built
from openly licensed clips downloaded from GitHub:
  speech   LibriSpeech excerpts (public domain), via github.com/librosa/data
  music    Creative Commons tracks from the same repo (drum & bass, electronic,
           string orchestra)
  effects  ESC-50 clips (door knock, glass breaking, footsteps, ...),
           via github.com/karolpiczak/ESC-50
Because every part is known, we can measure exactly how much of each one
survives in a model's output:

  music left   level of the music remaining in the output relative to the
               original music (dB; lower = less bleed)
  music gap    output level during a stretch with *only* music, relative to
               the music there (dB; lower = cleaner silence)
  sfx kept     sound-effect level in the output vs. original (dB; 0 = all kept)
  speech kept  same for speech (dB; 0 = all kept)
  artifacts    output energy that isn't any scaled/filtered source, relative
               to what should be kept (dB; lower = cleaner)

Contributions are estimated per STFT bin and 0.4 s block by least squares
(output ~= sum_i g_i * source_i), like BSS-Eval's allowed distortion.

Usage:  python bench/benchmark.py [--engine bandit|demucs|dnr_demucs] [--strengths 0,1,4]
                                 [--passes 1,2] [--seconds 14] [--save DIR]
"""

import argparse
import csv
import io
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import engines  # noqa: E402

SR = engines.SR
DATA = os.path.join(engines.CACHE, "bench")
LIBROSA = "https://raw.githubusercontent.com/librosa/data/main/audio/"
ESC = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/"
SPEECH = ["198-209-0000.ogg", "3436-172162-0000.ogg"]
MUSIC = {
    "drum&bass": "admiralbob77_-_Choice_-_Drum-bass.ogg",
    "electronic": "Kevin_MacLeod_-_Vibe_Ace.ogg",
    "orchestral": "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra).ogg",
}
SFX = ["door_wood_knock", "glass_breaking", "footsteps", "dog", "car_horn",
       "thunderstorm", "keyboard_typing", "siren", "helicopter"]


def fetch(url, name):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def load(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "2",
                          "-ar", str(SR), "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def rms_norm(x, db):
    active = x[:, np.abs(x).max(0) > 1e-3 * np.abs(x).max()]
    r = np.sqrt(np.mean(active ** 2)) + 1e-9
    return x * (10 ** (db / 20) / r)


def place(buf, clip, t0, t1):
    a, b = int(t0 * SR), int(t1 * SR)
    clip = clip[:, : b - a]
    fade = np.minimum(1, np.minimum(np.arange(clip.shape[1]), np.arange(clip.shape[1])[::-1]) / (0.01 * SR))
    buf[:, a:a + clip.shape[1]] += clip * fade


def build_mixes(seconds, music_db):
    speech = np.concatenate([load(fetch(LIBROSA + f, f)) for f in SPEECH], axis=1)
    speech = rms_norm(speech, -23)
    rows = list(csv.DictReader(io.StringIO(open(fetch(ESC + "meta/esc50.csv", "esc50.csv")).read())))
    sfx = {}
    for cat in SFX:
        f = next(r["filename"] for r in rows if r["category"] == cat)
        sfx[cat] = rms_norm(load(fetch(ESC + "audio/" + f, f)), -24)
    mixes = []
    L = int(seconds * SR)
    # Layout (fractions of the mix): speech | speech+sfx | music only | sfx only | speech | speech+sfx
    for k, (genre, mf) in enumerate(MUSIC.items()):
        music = load(fetch(LIBROSA + mf, mf))
        off = int(min(20 * SR, music.shape[1] - L - 1))
        music = rms_norm(music[:, off:off + L], -23 + music_db)
        sp = np.zeros((2, L), np.float32)
        fx = np.zeros((2, L), np.float32)
        s0 = k * int(5 * SR)
        place(sp, speech[:, s0:], 0, 0.35 * seconds)
        place(sp, speech[:, s0 + int(0.35 * seconds * SR):], 0.65 * seconds, seconds)
        cats = SFX[3 * k: 3 * k + 3]
        place(fx, sfx[cats[0]], 0.2 * seconds, 0.35 * seconds)   # over speech
        place(fx, sfx[cats[1]], 0.5 * seconds, 0.65 * seconds)   # alone (with music)
        place(fx, sfx[cats[2]], 0.8 * seconds, 0.95 * seconds)   # over speech
        gap = (int(0.37 * seconds * SR), int(0.48 * seconds * SR))  # music only
        mixes.append(dict(name=f"{genre} + {', '.join(cats)}", speech=sp, sfx=fx, music=music,
                          mix=sp + fx + music, gap=gap))
    return mixes


def stft(x):
    return torch.stft(torch.from_numpy(x), 2048, 512, window=torch.hann_window(2048), return_complex=True)


def measure(out, m):
    """Least-squares split of `out` into contributions of each known source."""
    srcs = [m["speech"], m["sfx"], m["music"]]
    S = torch.stack([stft(s) for s in srcs], -1)          # [C, F, N, 3]
    O = stft(out.astype(np.float32))                      # [C, F, N]
    blk = 32
    contrib = torch.zeros(3)
    resid = 0.0
    for a in range(0, O.shape[-1], blk):
        A = S[:, :, a:a + blk]                            # [C, F, n, 3]
        o = O[:, :, a:a + blk].unsqueeze(-1)              # [C, F, n, 1]
        AhA = A.conj().transpose(-1, -2) @ A
        ridge = 1e-6 * AhA.diagonal(dim1=-2, dim2=-1).real.sum(-1)[..., None, None] + 1e-10
        g = torch.linalg.solve(AhA + ridge * torch.eye(3), A.conj().transpose(-1, -2) @ o)
        parts = A * g.transpose(-1, -2)                   # [C, F, n, 3]
        contrib += (parts.abs() ** 2).sum((0, 1, 2))
        resid += float(((o.squeeze(-1) - parts.sum(-1)).abs() ** 2).sum())
    energy = torch.stack([(s.abs() ** 2).sum() for s in S.unbind(-1)])
    db = lambda a, b: 10 * np.log10(max(float(a), 1e-12) / max(float(b), 1e-12))
    g0, g1 = m["gap"]
    return {
        "music left": db(contrib[2], energy[2]),
        "music gap": db((out[:, g0:g1] ** 2).sum(), (m["music"][:, g0:g1] ** 2).sum()),
        "sfx kept": db(contrib[1], energy[1]),
        "speech kept": db(contrib[0], energy[0]),
        "artifacts": db(resid, energy[0] + energy[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=14)
    ap.add_argument("--music-db", type=float, default=-3, help="music level relative to speech")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--engine", default="bandit", choices=sorted(engines.ENGINES))
    ap.add_argument("--strengths", default="0,0.5,1,2,4", help="bleed-suppression strengths to compare")
    ap.add_argument("--passes", default="1", help="model passes to compare, e.g. 1,2")
    ap.add_argument("--save", help="directory to write outputs as WAV for listening")
    args = ap.parse_args()
    strengths = [float(v) for v in args.strengths.split(",")]
    passes = sorted(int(v) for v in args.passes.split(","))
    mixes = build_mixes(args.seconds, args.music_db)
    eng = engines.ENGINES[args.engine]().to(args.device)
    pick = lambda st, names: sum(st[n] for n in names)

    rows = {(p, s): [] for p in passes for s in strengths}
    secs = 0.0
    for m in mixes:
        x = m["mix"]
        for p in range(1, passes[-1] + 1):
            t = time.time()
            stems = eng.stems_of(x)  # strengths only re-mask, so the model runs once per pass
            if p == 1:
                secs += time.time() - t
            x = pick(stems, eng.keep_stems)
            if p not in passes:
                continue
            for s in strengths:
                out = eng.apply(stems, s).cpu().numpy()  # same filtering as the server
                rows[p, s].append(measure(out, m))
                if args.save:
                    import soundfile as sf
                    os.makedirs(args.save, exist_ok=True)
                    tag = m["name"].split(" ")[0].replace("&", "n")
                    sf.write(os.path.join(args.save, f"{args.engine}_p{p}_s{s:g}_{tag}.wav"), out.T, SR)
                    sf.write(os.path.join(args.save, f"_mix_{tag}.wav"), m["mix"].T, SR)
    speed = secs / (len(mixes) * args.seconds)

    print(f"\n{args.engine} on {args.device}: {speed:.2f} s of compute per second of audio per pass\n")
    print("| passes | bleed suppression | music left dB ↓ | music-only gap dB ↓ | SFX kept dB (0 best) "
          "| speech kept dB (0 best) | artifacts dB ↓ |")
    print("|---|---|---|---|---|---|---|")
    for (p, s), rs in rows.items():
        a = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        print(f"| {p} | {s:g}{' (off)' if s == 0 else ''} | {a['music left']:.1f} | {a['music gap']:.1f} "
              f"| {a['sfx kept']:.1f} | {a['speech kept']:.1f} | {a['artifacts']:.1f} |")
    print("\nPer mix:")
    for (p, s), rs in rows.items():
        for m, r in zip(mixes, rs):
            print(f"  p={p} s={s:<4g} {m['name']:55s} " + "  ".join(f"{k} {v:6.1f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
