import os
import argparse
import numpy as np
import pandas as pd

from features import load_wav, extract_features

# ---------------------------------------------------------
# Resolve paths relative to THIS file, not the terminal
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# dataset.py is inside starter/
# eot_data is beside starter/
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "eot_data")


def process_folder(data_dir):

    labels_path = os.path.join(data_dir, "labels.csv")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"\nlabels.csv not found!\nExpected:\n{labels_path}"
        )

    labels = pd.read_csv(labels_path)

    audio_cache = {}

    X = []
    y = []
    groups = []

    turn_ids = []
    pause_indices = []
    audio_files = []

    for _, row in labels.iterrows():

        audio_path = os.path.join(
            data_dir,
            row["audio_file"]
        )

        if audio_path not in audio_cache:
            audio_cache[audio_path] = load_wav(audio_path)

        x, sr = audio_cache[audio_path]

        feature = extract_features(
            x,
            sr,
            float(row["pause_start"])
        )

        label = 1 if row["label"] == "eot" else 0

        X.append(feature)
        y.append(label)
        groups.append(row["turn_id"])

        turn_ids.append(row["turn_id"])
        pause_indices.append(int(row["pause_index"]))
        audio_files.append(audio_path)

    return (
        np.array(X),
        np.array(y),
        np.array(groups),
        np.array(turn_ids),
        np.array(pause_indices),
        np.array(audio_files),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        default="dataset.npz"
    )

    args = parser.parse_args()

    X_all = []
    y_all = []
    groups_all = []

    turn_all = []
    pause_all = []
    file_all = []

    for lang in ["english", "hindi"]:

        folder = os.path.join(BASE_DIR, lang)

        print(f"Processing {folder}")

        (
            X,
            y,
            groups,
            turns,
            pauses,
            files
        ) = process_folder(folder)

        X_all.append(X)
        y_all.append(y)
        groups_all.append(groups)

        turn_all.append(turns)
        pause_all.append(pauses)
        file_all.append(files)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    groups_all = np.concatenate(groups_all)

    turn_all = np.concatenate(turn_all)
    pause_all = np.concatenate(pause_all)
    file_all = np.concatenate(file_all)

    np.savez(
        args.out,
        X=X_all,
        y=y_all,
        groups=groups_all,
        turn_id=turn_all,
        pause_index=pause_all,
        audio_file=file_all,
    )

    print("\n===============================")
    print("Dataset Created Successfully")
    print("===============================")

    print("Samples :", len(y_all))
    print("Features:", X_all.shape[1])

    print("\nSaved to :", args.out)


if __name__ == "__main__":
    main()