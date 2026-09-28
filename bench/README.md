# Benchmark

`benchmark.py` builds test mixes where every part is known (speech + non-music sound
effects + background music), runs BandIt Plus on them, and measures how much of each part
survives. Because the ground truth is known, "bleed" is measured directly instead of judged
by ear.

```bash
python bench/benchmark.py                       # CPU or GPU, auto
python bench/benchmark.py --strengths 0,1,4 --save out/   # also write WAVs to listen to
```

Test audio is downloaded from GitHub on first run (nothing is committed):

* speech: LibriSpeech excerpts (public domain), from [librosa/data](https://github.com/librosa/data)
* music: Creative Commons tracks from the same repo (drum & bass, electronic, string orchestra)
* sound effects: [ESC-50](https://github.com/karolpiczak/ESC-50) clips (door knock, glass
  breaking, footsteps, dog, car horn, thunderstorm, keyboard typing, siren, helicopter)

Each 14 s mix has speech, effects over speech, a music-only stretch, and effects with no speech.
Music is 3 dB below speech (loud background music).

## Metrics

| Metric | Meaning | Ideal |
|---|---|---|
| music left | level of the original music still in the output | as low as possible |
| music gap | output level during the music-only stretch | as low as possible |
| SFX kept | sound-effect level in the output vs. the original | 0 dB |
| speech kept | speech level in the output vs. the original | 0 dB |
| artifacts | output that isn't any of the sources | as low as possible |

Each source's contribution is estimated per frequency bin and 0.4 s block by least squares
(similar to BSS-Eval). Sanity check: a perfect output scores −91 dB music left and 0 dB
for SFX and speech. Adding 10% of the music back scores exactly −20 dB.

## Results (4-core CPU, 3 mixes)

### Bleed suppression

The model runs once. The strength only changes the extra spectral mask
(`engines.suppress_bleed`). Speed: 6.1 s of CPU time per second of audio.

| Strength | Music left | Music gap | SFX kept | Speech kept | Artifacts |
|---|---|---|---|---|---|
| 0 (off) | −19.5 | −103.1 | −1.1 | −0.8 | −21.0 |
| 0.5 | −20.4 | −144.9 | −1.2 | −0.9 | −20.9 |
| **1 (default)** | −20.6 | −144.9 | −1.3 | −1.0 | −20.8 |
| 2 | −20.9 | −144.9 | −1.3 | −1.1 | −20.5 |
| 4 | −21.2 | −144.9 | −1.4 | −1.2 | −20.2 |

### Why not a vocal model

An earlier run compared BandIt Plus with the vocal-separation models that are popular for
"bleedless" results. Keeping the vocal stem removes a little more music, but it also removes
almost all sound effects, and those models are much slower:

| Remover | Music left | SFX kept | Speech kept | CPU s per audio s |
|---|---|---|---|---|
| BandIt Plus (speech + effects) | −19.5 | −1.1 | −0.8 | 5.1 |
| BS-RoFormer (viperx 1297), vocal stem | −26.8 | **−26.3** | −0.1 | 17.8 |
| BandIt Plus + RoFormer singing rescue | −19.5 | −1.1 | −0.1 | 17.7 |

Those models were removed from the server. BandIt v2 (the multilingual DnR v3 successor) has
the same network size, so it would be no faster; the higher-rated MVSep DnR models aren't
publicly downloadable.
