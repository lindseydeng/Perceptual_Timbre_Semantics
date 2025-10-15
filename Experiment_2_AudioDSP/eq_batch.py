'''
python implementation of https://audealize.appspot.com/static/js/effects/equalizer.js

'''


import numpy as np
import scipy.signal as signal
import os
import json
import argparse
import soundfile as sf
from pydub import AudioSegment
import tempfile


def create_peaking_filter(fs, f0, Q, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def apply_equalizer(data, fs, gains, freqs=None, Q=4.31, gain_range_db=12):
    """
    Apply a 40-band EQ using peaking filters.

    Parameters:
        - data: np.ndarray, shape (samples, channels)
        - fs: sample rate
        - gains: 40-band gain values in normalized scale (e.g., -1 to 1)
        - freqs: list of center frequencies
        - Q: quality factor
        - gain_range_db: maximum gain to scale to (default ±12 dB)
    """
    if freqs is None:
        freqs = [20, 50, 83, 120, 161, 208, 259, 318, 383, 455,
                 537, 628, 729, 843, 971, 1114, 1273, 1452, 1652, 1875,
                 2126, 2406, 2719, 3070, 3462, 3901, 4392, 4941, 5556, 6244,
                 7014, 7875, 8839, 9917, 11124, 12474, 13984, 15675, 17566, 19682]

    output = data.copy()
    # Scale normalized gains to decibel
    gains_db = [g * gain_range_db for g in gains]

    for gain_db, f0 in zip(gains_db, freqs):
        if f0 >= fs / 2:
            continue
        b, a = create_peaking_filter(fs, f0, Q, gain_db)
        output = signal.lfilter(b, a, output, axis=0)

    return output


def convert_mp3_to_wav(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name

def load_descriptor_parameters(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    descriptor_map = {}
    for entry in data:
        word = entry['word']
        if word not in descriptor_map:
            descriptor_map[word] = []
        descriptor_map[word].append(entry['settings'])
    return descriptor_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply EQ manipulation to an audio file.")
    parser.add_argument("input", help="Input WAV or MP3 file")
    parser.add_argument("output_dir", help="Directory to save EQ'd files")
    parser.add_argument("--eq_json", default="filtered_eqpoints.json", help="EQ descriptor JSON file")
   # parser.add_argument("--amounts", default="0.3,0.6,1.0", help="Comma-separated list of EQ amounts")
    parser.add_argument("--amounts", help="Comma-separated list of EQ amounts (e.g., 0.1,0.2,...)")

    args = parser.parse_args()

    # default amount list if not provided
    if args.amounts is None:
        amounts = [round(x, 2) for x in np.arange(0.1, 1.01, 0.1)]
    else:
        amounts = [float(a) for a in args.amounts.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    eq_params = load_descriptor_parameters(args.eq_json)

    input_path = args.input
    if input_path.lower().endswith(".mp3"):
        print(f"🔄 Converting MP3 to WAV: {input_path}")
        input_path = convert_mp3_to_wav(input_path)

    data, fs = sf.read(input_path)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    # EQ Processing
    for word, settings_list in eq_params.items():
        avg_gains = np.mean(settings_list, axis=0)
        for amount in amounts:
            scaled_gains = [g * amount for g in avg_gains]
            eq_data = apply_equalizer(data, fs, scaled_gains)
            output_path = os.path.join(args.output_dir, f"{base_name}_eq_{word}_{amount}.wav")
            sf.write(output_path, eq_data, fs)
            print(f"Saved EQ: {output_path}")

    print(f"\n All EQ-manipulated files saved to: {args.output_dir}")