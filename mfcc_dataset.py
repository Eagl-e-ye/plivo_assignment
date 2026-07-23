"""
Build MFCC dataset for CNN training.

Directory structure:

eot_data/
│
├── english/
│   ├── labels.csv
│   └── audio...
│
└── hindi/
    ├── labels.csv
    └── audio...

Output:
    mfcc_dataset.npz
"""

import os
import csv
import argparse

import librosa
import numpy as np

from features import load_wav, speech_before

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

WINDOW_SECONDS = 1.5

N_MFCC = 40

N_FFT = 1024

HOP_LENGTH = 160

MAX_FRAMES = 150


# -------------------------------------------------------
# MFCC Extraction
# -------------------------------------------------------

def compute_mfcc(seg, sr):
    """
    Returns a normalized MFCC of shape (40,150)
    """

    # Empty audio
    if len(seg) == 0:
        return np.zeros((N_MFCC, MAX_FRAMES), dtype=np.float32)

    mfcc = librosa.feature.mfcc(
        y=seg,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Per-sample normalization
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

    # Pad / Crop
    if mfcc.shape[1] < MAX_FRAMES:

        pad = MAX_FRAMES - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, pad)),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :MAX_FRAMES]

    return mfcc.astype(np.float32)


# -------------------------------------------------------
# Single language
# -------------------------------------------------------

def load_language(folder):

    csv_path = os.path.join(folder, "labels.csv")

    rows = list(csv.DictReader(open(csv_path)))

    cache = {}

    X = []
    y = []
    groups = []

    for row in rows:

        wav_path = os.path.join(
            folder,
            row["audio_file"]
        )

        if wav_path not in cache:
            cache[wav_path] = load_wav(wav_path)

        audio, sr = cache[wav_path]

        seg = speech_before(
            audio,
            sr,
            float(row["pause_start"]),
            WINDOW_SECONDS
        )

        mfcc = compute_mfcc(seg, sr)

        # Channel dimension
        mfcc = mfcc[np.newaxis, :, :]

        X.append(mfcc)

        y.append(
            1 if row["label"] == "eot" else 0
        )

        groups.append(row["turn_id"])

    return (
        np.stack(X),
        np.array(y, dtype=np.int64),
        np.array(groups)
    )


# -------------------------------------------------------
# Both English and Hindi
# -------------------------------------------------------

def load_all(base_dir):

    X_all = []

    y_all = []

    groups_all = []

    for language in ["english", "hindi"]:

        print(f"Loading {language}...")

        folder = os.path.join(
            base_dir,
            language
        )

        X, y, groups = load_language(folder)

        X_all.append(X)

        y_all.append(y)

        groups_all.append(groups)

    X = np.concatenate(X_all, axis=0)

    y = np.concatenate(y_all)

    groups = np.concatenate(groups_all)

    return X, y, groups


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root folder containing english/ and hindi/"
    )

    parser.add_argument(
        "--output",
        default="mfcc_dataset.npz"
    )

    args = parser.parse_args()

    X, y, groups = load_all(args.data_dir)

    print()

    print("=" * 50)

    print("Dataset Summary")

    print("=" * 50)

    print("Samples :", len(y))

    print("Shape   :", X.shape)

    print("Labels  :", y.shape)

    print("Groups  :", groups.shape)

    print()

    np.savez_compressed(

        args.output,

        X=X,

        y=y,

        groups=groups

    )

    print(f"Saved dataset -> {args.output}")


if __name__ == "__main__":
    main()