"""
Prediction using

Extra Trees
+
CNN

Final probability

P = w_tree * P_tree + w_cnn * P_cnn
"""

import os
import csv
import argparse

import joblib
import librosa
import numpy as np

import torch
import torch.nn as nn

from features import (
    load_wav,
    speech_before,
    extract_features
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

############################################################
# Configuration
############################################################

WINDOW_SECONDS = 1.5

N_MFCC = 40

N_FFT = 1024

HOP_LENGTH = 160

MAX_FRAMES = 150

TREE_WEIGHT = 0.30

CNN_WEIGHT = 0.70


############################################################
# CNN
############################################################

class CNNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(128, 64),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(64, 1)

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x.squeeze(1)


############################################################
# Load Models
############################################################

def load_models(tree_path, cnn_path):

    tree_model = joblib.load(tree_path)

    device = torch.device(

        "mps"

        if torch.backends.mps.is_available()

        else "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    cnn = CNNModel().to(device)

    cnn.load_state_dict(

        torch.load(
            cnn_path,
            map_location=device
        )

    )

    cnn.eval()

    return tree_model, cnn, device


############################################################
# MFCC
############################################################

def compute_mfcc(seg, sr):

    if len(seg) == 0:

        return np.zeros(
            (1, N_MFCC, MAX_FRAMES),
            dtype=np.float32
        )

    mfcc = librosa.feature.mfcc(

        y=seg,

        sr=sr,

        n_mfcc=N_MFCC,

        n_fft=N_FFT,

        hop_length=HOP_LENGTH

    )

    mfcc = (

        mfcc - mfcc.mean()

    ) / (

        mfcc.std() + 1e-8

    )

    if mfcc.shape[1] < MAX_FRAMES:

        pad = MAX_FRAMES - mfcc.shape[1]

        mfcc = np.pad(

            mfcc,

            ((0, 0), (0, pad))

        )

    else:

        mfcc = mfcc[:, :MAX_FRAMES]

    mfcc = mfcc[np.newaxis]

    return mfcc.astype(np.float32)


############################################################
# Extra Trees Prediction
############################################################

def predict_tree(
    tree_model,
    audio,
    sr,
    pause_start
):
    """
    Returns P(EOT) from the handcrafted-feature model.
    """

    feat = extract_features(
        audio,
        sr,
        pause_start,
        WINDOW_SECONDS
    )

    feat = feat.reshape(1, -1)

    prob = tree_model.predict_proba(feat)[0, 1]

    return float(prob)


############################################################
# CNN Prediction
############################################################

@torch.no_grad()
def predict_cnn(
    cnn,
    device,
    audio,
    sr,
    pause_start
):
    """
    Returns P(EOT) from the CNN.
    """

    seg = speech_before(
        audio,
        sr,
        pause_start,
        WINDOW_SECONDS
    )

    mfcc = compute_mfcc(
        seg,
        sr
    )

    x = torch.from_numpy(
        mfcc
    ).unsqueeze(0)

    # Shape:
    # (1,1,40,150)

    x = x.to(device)

    logits = cnn(x)

    prob = torch.sigmoid(
        logits
    ).item()

    return float(prob)


############################################################
# Ensemble
############################################################

def ensemble_probability(
    p_tree,
    p_cnn,
    tree_weight=TREE_WEIGHT,
    cnn_weight=CNN_WEIGHT
):
    """
    Weighted average of the two models.
    """

    return (
        tree_weight * p_tree +
        cnn_weight * p_cnn
    )


############################################################
# Predict One Pause
############################################################

def predict_pause(
    tree_model,
    cnn,
    device,
    audio,
    sr,
    pause_start
):
    """
    Predict probability for one pause.
    """

    p_tree = predict_tree(
        tree_model,
        audio,
        sr,
        pause_start
    )

    p_cnn = predict_cnn(
        cnn,
        device,
        audio,
        sr,
        pause_start
    )

    p_final = ensemble_probability(
        p_tree,
        p_cnn
    )

    return (
        p_tree,
        p_cnn,
        p_final
    )


############################################################
# Predict Dataset
############################################################

def predict_dataset(
    data_dir,
    tree_model,
    cnn,
    device,
    output_csv
):

    csv_path = os.path.join(
        data_dir,
        "labels.csv"
    )

    rows = list(
        csv.DictReader(
            open(csv_path)
        )
    )

    cache = {}

    predictions = []

    print()

    print("Predicting...")
    true_labels = []

    tree_probs = []

    cnn_probs = []

    final_probs = []

    for i, row in enumerate(rows):

        wav_path = os.path.join(
            data_dir,
            row["audio_file"]
        )

        ###############################################
        # Cache audio
        ###############################################

        if wav_path not in cache:

            cache[wav_path] = load_wav(
                wav_path
            )

        audio, sr = cache[wav_path]

        ###############################################
        # Predict
        ###############################################

        p_tree, p_cnn, p_final = predict_pause(
            tree_model,
            cnn,
            device,
            audio,
            sr,
            float(row["pause_start"])
        )

        tree_probs.append(p_tree)
        cnn_probs.append(p_cnn)
        final_probs.append(p_final)

        true_labels.append(
            1 if row["label"] == "eot" else 0
        )

        prob = p_final

    ###############################################
    # Save CSV
    ###############################################

    with open(
        output_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(

            [
                "turn_id",
                "pause_index",
                "p_eot"
            ]

        )

        writer.writerows(predictions)

    print()

    print(f"Saved predictions -> {output_csv}")
    ########################################################
    # Evaluation
    ########################################################

    tree_pred = (np.array(tree_probs) >= 0.5).astype(int)
    cnn_pred = (np.array(cnn_probs) >= 0.5).astype(int)
    final_pred = (np.array(final_probs) >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)

    print("\nExtra Trees")
    print(f"Accuracy : {accuracy_score(true_labels, tree_pred):.4f}")
    print(f"AUC      : {roc_auc_score(true_labels, tree_probs):.4f}")
    print(f"F1       : {f1_score(true_labels, tree_pred):.4f}")

    print("\nCNN")
    print(f"Accuracy : {accuracy_score(true_labels, cnn_pred):.4f}")
    print(f"AUC      : {roc_auc_score(true_labels, cnn_probs):.4f}")
    print(f"F1       : {f1_score(true_labels, cnn_pred):.4f}")

    print("\nEnsemble")
    print(f"Accuracy : {accuracy_score(true_labels, final_pred):.4f}")
    print(f"AUC      : {roc_auc_score(true_labels, final_probs):.4f}")
    print(f"F1       : {f1_score(true_labels, final_pred):.4f}")


############################################################
# Main
############################################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--data_dir",

        required=True,

        help="english/ or hindi/ folder"

    )

    parser.add_argument(

        "--tree_model",

        default="model.pkl"

    )

    parser.add_argument(

        "--cnn_model",

        default="cnn_model.pt"

    )

    parser.add_argument(

        "--output",

        default="predictions.csv"

    )

    parser.add_argument(

        "--tree_weight",

        type=float,

        default=0.30

    )

    parser.add_argument(

        "--cnn_weight",

        type=float,

        default=0.70

    )

    args = parser.parse_args()

    ########################################################
    # Update global weights
    ########################################################

    global TREE_WEIGHT
    global CNN_WEIGHT

    TREE_WEIGHT = args.tree_weight
    CNN_WEIGHT = args.cnn_weight

    ########################################################
    # Load models
    ########################################################

    tree_model, cnn, device = load_models(

        args.tree_model,

        args.cnn_model

    )

    ########################################################
    # Predict
    ########################################################

    for lang in ["english", "hindi"]:

        folder = os.path.join(args.data_dir, lang)

        output = f"{lang}_predictions.csv"

        print(f"\nRunning on {lang}...\n")

        predict_dataset(
            folder,
            tree_model,
            cnn,
            device,
            output
        )


if __name__ == "__main__":
    main()