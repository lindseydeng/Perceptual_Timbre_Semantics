
'''
python implementation of https://audealize.appspot.com/static/js/effects/reverb.js
'''
import argparse
import json
import os
import re
import numpy as np
import soundfile as sf
import time
from scipy.signal import butter, lfilter


# --------- Audealize reverb core ---------

def prev_prime(n: int) -> int:
    if n < 2:
        return 2
    def is_prime(k: int) -> bool:
        if k % 2 == 0:
            return k == 2
        i, r = 3, int(k ** 0.5)
        while i <= r:
            if k % i == 0:
                return False
            i += 2
        return True
    k = int(n)
    while k >= 2 and not is_prime(k):
        k -= 1
    return max(2, k)

def delay_samples(seconds: float, sr: int) -> int:
    return max(0, int(np.floor(seconds * sr + 1e-9)))

def process_filter(x: np.ndarray, D: int, g1: float, g2: float) -> np.ndarray:
    """
    JS Filter node:
        y[n] = g1 * x[n] + x[n - D] + g2 * y[n - D]
    """
    y = np.zeros_like(x, dtype=np.float32)
    if D <= 0:
        y = (g1 + 1.0) * x.astype(np.float32)
        return y
    for n in range(len(x)):
        x_d = x[n - D] if n >= D else 0.0
        y_d = y[n - D] if n >= D else 0.0
        y[n] = g1 * x[n] + x_d + g2 * y_d
    return y

'''
def one_pole_lowpass(x: np.ndarray, fc: float, sr: int) -> np.ndarray:
    """
    Simple 1-pole low-pass. Swap with a biquad if you want closer to WebAudio.
    """
    if fc <= 0:
        return np.zeros_like(x, dtype=np.float32)
    if fc >= sr * 0.49:
        return x.astype(np.float32, copy=True)
    dt = 1.0 / sr
    RC = 1.0 / (2.0 * np.pi * fc)
    alpha = dt / (RC + dt)
    y = np.zeros_like(x, dtype=np.float32)
    yp = 0.0
    for n in range(len(x)):
        yp = yp + alpha * (float(x[n]) - yp)
        y[n] = yp
    return y
'''

def biquad_lowpass(x: np.ndarray, fc: float, sr: int, order: int = 2) -> np.ndarray:
    """
    Apply a biquad low-pass filter to the input signal x.
    Equivalent to WebAudio BiquadFilterNode with type='lowpass'.
    """
    nyq = 0.5 * sr
    normal_cutoff = min(fc / nyq, 0.9999)  # prevent instability
    b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, x).astype(np.float32)

def audealize_reverb(
    audio: np.ndarray,
    sr: int,
    d: float, g: float, m: float, f: float, E: float,
    wetdry: float = 1.0,
    mindelay: float = 0.01,
    allpass_gain: float = 0.1
) -> np.ndarray:
    """
    Offline rendition of audealize/static/js/effects/reverb.js.
    Returns stereo float32 audio.
    """
    x = audio.astype(np.float32)
    if x.ndim == 2 and x.shape[1] == 2:
        x_mono = x.mean(axis=1)
        dry_L, dry_R = x[:, 0], x[:, 1]
    elif x.ndim == 1:
        x_mono = x
        dry_L = dry_R = x
    else:
        raise ValueError("audio must be mono (N,) or stereo (N,2)")

    N = len(x_mono)

    # Wet/dry cosine crossfade
    wetdry = float(np.clip(wetdry, 0.0, 1.0))
    wet_gain = np.cos((1.0 - wetdry) * 0.5 * np.pi)
    dry_gain = np.cos(wetdry * 0.5 * np.pi)

    # Comb bank (6 parallel), unify RT via gain per comb
    g = float(np.clip(g, 1e-6, 0.9999))
    d = float(max(1e-4, d))
    rt = d * (np.log(0.001) / np.log(g))

    comb_sum = np.zeros(N, dtype=np.float32)
    for i in range(6):
        delay_sec = d * (15 - i) / 15.0
        gain_i = float(np.power(0.001, delay_sec / rt))
        D = prev_prime(delay_samples(delay_sec, sr))
        comb_sum += process_filter(x_mono, D, g1=0.0, g2=gain_i)

    # Allpass (stereo decorrelation)
    da = mindelay + 0.006
    D_left  = prev_prime(delay_samples(max(0.0, da + 0.5 * m), sr))
    D_right = prev_prime(delay_samples(max(0.0, da - 0.5 * m), sr))
    ap_L = process_filter(comb_sum, D_left,  g1=-allpass_gain, g2=allpass_gain)
    ap_R = process_filter(comb_sum, D_right, g1=-allpass_gain, g2=allpass_gain)

    # Lowpass per channel
    #lp_L = one_pole_lowpass(ap_L, f, sr)
    #lp_R = one_pole_lowpass(ap_R, f, sr)
    lp_L = biquad_lowpass(ap_L, f, sr)
    lp_R = biquad_lowpass(ap_R, f, sr)

    # Early path (mindelay) + energy mixing
    D_min = delay_samples(mindelay, sr)
    early = np.zeros_like(x_mono)
    if D_min > 0:
        early[D_min:] = x_mono[:-D_min]
    else:
        early = x_mono.copy()

    totalGain = E + 1.0
    g1_energy = 1.0 / totalGain
    gainclean = np.cos((1.0 - g1_energy) * 0.125 * np.pi)
    gainlate  = np.cos(g1_energy * 0.375 * np.pi)
    gainscale = 0.5 * 0.8 / (gainclean + gainlate + 1e-12)

    wet_L = gainscale * (gainlate * lp_L + gainclean * early)
    wet_R = gainscale * (gainlate * lp_R + gainclean * early)

    y_L = dry_gain * dry_L + wet_gain * wet_L
    y_R = dry_gain * dry_R + wet_gain * wet_R
    y = np.stack([y_L, y_R], axis=1).astype(np.float32)
    return np.clip(y, -1.0, 1.0)


# --------- JSON / batch driver ---------

def load_words(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = []
    for item in data:
        if not isinstance(item, dict):
            continue
        w = str(item.get("word", "")).strip()
        st = item.get("settings", None)
        if not w or not isinstance(st, list) or len(st) != 5:
            continue
        words.append((w, st))
    return words

def sanitize_for_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s

def main():
    p = argparse.ArgumentParser(description="Batch render Audealize-style reverb for multiple words and wet/dry levels.")
    p.add_argument("input", help="input audio (wav/flac/aiff)")
    p.add_argument("json", help="JSON file with [{word, settings:[d,g,m,f,E]}, ...]")
    p.add_argument("--outdir", required=True,
                   help="Output folder for processed WAV files (will be created if missing)")
    #p.add_argument("--wetdry", nargs="+", type=float, default=[0.3, 0.6, 1.0],
                   #help="List of wet/dry values 0..1 (default: 0.3 0.6 1.0)")
    p.add_argument("--wetdry", nargs="+", type=float,
               help="List of wet/dry values 0..1 (e.g., 0.3 0.6 1.0). If not specified, uses 0.1 to 1.0 in 0.1 steps.")
    p.add_argument("--mindelay", type=float, default=0.01, help="Early/clean path delay (s), default 0.01")
    p.add_argument("--f-normalized", action="store_true",
                   help="Interpret f as normalized 0..1 and scale to Nyquist")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N words from JSON (optional)")
    args = p.parse_args()

    # default wetdry list if not provided
    if args.wetdry is None:
        args.wetdry = [round(wd, 2) for wd in np.arange(0.1, 1.01, 0.1)]


    # create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # load audio
    audio, sr = sf.read(args.input, always_2d=False)
    if getattr(audio, "dtype", None) is not None and audio.dtype.kind in "iu":
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    audio = audio.astype(np.float32)

    base = os.path.splitext(os.path.basename(args.input))[0]  # e.g., 'guitar'

    # load words
    entries = load_words(args.json)
    if args.limit is not None:
        entries = entries[:args.limit]
    if not entries:
        raise SystemExit("No valid {word, settings} entries found in JSON.")

    wetdry_list = [float(np.clip(wd, 0.0, 1.0)) for wd in args.wetdry]

    # Helpful header prints
    ch = 1 if audio.ndim == 1 else audio.shape[1]
    print(f"Input: {args.input}  sr={sr}  channels={ch}")
    print(f"Output dir: {args.outdir}")
    print(f"Words loaded: {len(entries)}  Wet/Dry levels: {wetdry_list}  f-normalized={args.f_normalized}")

    total = len(entries) * len(wetdry_list)
    count = 0

    for word, settings in entries:
        d, g, m, f, E = settings
        f_hz = float(f) * (sr / 2.0) if args.f_normalized else float(f)
        word_tag = sanitize_for_filename(word)

        for wd in wetdry_list:
            t0 = time.perf_counter()
            out = audealize_reverb(
                audio=audio, sr=sr,
                d=float(d), g=float(g), m=float(m), f=f_hz, E=float(E),
                wetdry=wd, mindelay=float(args.mindelay)
            )
            out_name = f"{base}_reverb_{word_tag}_{wd:.1f}.wav"
            out_path = os.path.join(args.outdir, out_name)
            sf.write(out_path, out, sr)
            count += 1
            dt = time.perf_counter() - t0
            print(f"[{count}/{total}] Wrote {out_path} ({dt:.2f}s)", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

