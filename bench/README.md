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

### Demucs

HTDemucs fine-tuned for vocals (`--engine demucs`). It keeps the "vocals" stem, and its
"other" stem drives the filter. Speed: 1.4–1.5 s of CPU time per second of audio.

| Passes | Strength | Music left | Music gap | SFX kept | Speech kept | Artifacts |
|---|---|---|---|---|---|---|
| 1 | 0 (off) | −22.1 | −53.9 | −14.4 | −0.1 | −22.0 |
| 1 | 1 | −23.4 | −61.2 | −15.8 | −0.3 | −21.4 |
| 1 | 4 | −24.1 | −66.6 | −16.4 | −0.5 | −20.7 |
| 1 | 8 | −24.5 | −69.8 | −16.9 | −0.6 | −20.3 |
| 1 | **16 (Normal)** | −24.9 | −73.0 | −17.4 | −0.8 | −20.0 |
| 1 | 32 | −25.3 | −75.9 | −18.1 | −1.0 | −19.7 |
| 1 | **64 (Strong)** | −25.7 | −78.5 | −18.9 | −1.3 | −19.4 |
| 2 | 0 (off) | −22.2 | −56.0 | −14.6 | −0.2 | −21.9 |
| 2 | 8 | −22.2 | −61.2 | −14.7 | −0.2 | −22.0 |

Each doubling of strength lowers the bleed by about 0.4 dB and costs about 0.2–0.3 dB of
speech. A second pass (running Demucs again on its own vocals) doesn't lower the bleed, and it
weakens the filter, because the second pass sees almost no music. So passes are not offered.

### UVR Voc FT and MelBand RoFormer

Two voice-only models from the UVR community (`--engine voc_ft`, `--engine melband`). For both,
"other" is the mix minus the vocals, and it drives the filter.

| Model | Strength | Music left | Music gap | SFX kept | Speech kept | Artifacts | CPU s / audio s |
|---|---|---|---|---|---|---|---|
| Voc FT | 0 (off) | −24.8 | −46.6 | −20.4 | −0.2 | −22.2 | 2.3 |
| Voc FT | 4 | −26.5 | −59.3 | −23.6 | −0.6 | −21.7 | |
| Voc FT | **16 (Normal)** | −26.9 | −69.0 | −25.0 | −1.0 | −21.1 | |
| Voc FT | **64 (Strong)** | −27.3 | −80.1 | −26.5 | −1.6 | −20.5 | |
| MelBand | 0 (off) | −26.9 | −116.2 | −27.6 | −0.2 | −23.3 | 11.3 |
| MelBand | **4 (Normal)** | −28.1 | −144.9 | −29.2 | −0.6 | −22.5 | |
| MelBand | **16 (Strong)** | −28.0 | −144.9 | −30.3 | −1.1 | −21.5 | |
| MelBand | 64 | −27.8 | −144.9 | −31.4 | −1.8 | −20.6 | |

MelBand stops improving at strength 4, so its Strong level mainly trades a little speech for
removing slightly more of whatever is left.

Validation against audio-separator's own implementation, on the same mix:

* MelBand: outputs agree to 29 dB, with identical bleed numbers.
* Voc FT: outputs agree to 27 dB after volume matching. audio-separator multiplies its output
  by the input's peak level, which makes its speech look 2.8 dB quieter. Ours keeps the
  original volume.

Voc FT runs slower here than through audio-separator's onnxruntime path, which measured
0.94 s/audio s on CPU. We convert the ONNX model to PyTorch so it runs on every GPU backend,
and we use 2× window overlap.

### DnR Demucs and Voc FT + DnR Demucs (AMD RX 9060 XT, ROCm)

Measured on a user's machine (the test machine couldn't reach Zenodo for DnR's weights). DnR
Demucs falls back to fp32 on this GPU (fp16 gives non-finite output).

| Model | Strength | Music left | Music gap | SFX kept | Speech kept | Artifacts |
|---|---|---|---|---|---|---|
| DnR Demucs | 0 (off) | −16.7 | −46.9 | −1.2 | −0.4 | −19.2 |
| DnR Demucs | **1 (Normal)** | −18.5 | −56.0 | −1.5 | −0.5 | −18.5 |
| DnR Demucs | **4 (Strong)** | −19.6 | −60.0 | −1.8 | −0.6 | −18.0 |
| Voc FT + DnR | 0 (off) | −17.0 | −44.7 | −0.4 | −0.1 | −18.7 |
| Voc FT + DnR | 1 | −18.8 | −48.1 | −0.6 | −0.3 | −18.0 |
| Voc FT + DnR | 4 | −19.8 | −48.7 | −0.9 | −0.5 | −17.4 |

Speed: Voc FT + DnR 0.56 s/audio s. DnR alone printed 1.68 s/audio s, but that run included
ROCm compiling kernels on first use.

The combined rows used a single filter driven by DnR's music estimate. It barely helped in
pauses (−48 dB), because the music Voc FT lets through isn't in DnR's estimate. The engine now
filters each part with its own model's estimate: voices at 4× the strength with Voc FT's
"other", effects with DnR's music. Normal is now effects 4 / voices 16. Re-run
`--engine voc_ft_dnr` for current numbers.

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
