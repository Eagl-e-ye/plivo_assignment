

import argparse
import csv
import os
import joblib
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from features import load_wav, extract_features


import argparse
import csv
import os
import joblib
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from features import load_wav, extract_features


############################################################
# Load Dataset
############################################################

def load_dataset(data_dir):

    rows = list(csv.DictReader(
        open(os.path.join(data_dir, "labels.csv"))
    ))

    cache = {}

    X = []
    y = []
    groups = []
    keys = []

    for r in rows:

        path = os.path.join(
            data_dir,
            r["audio_file"]
        )

        if path not in cache:
            cache[path] = load_wav(path)

        x, sr = cache[path]

        feat = extract_features(
            x,
            sr,
            float(r["pause_start"])
        )

        X.append(feat)

        y.append(
            1 if r["label"] == "eot" else 0
        )

        groups.append(r["turn_id"])

        keys.append(
            (
                r["turn_id"],
                r["pause_index"]
            )
        )

    return (
        np.array(X),
        np.array(y),
        np.array(groups),
        keys
    )


def load_all_datasets(base_dir):

    X_all = []
    y_all = []
    groups_all = []
    keys_all = []

    for lang in ["english", "hindi"]:

        folder = os.path.join(base_dir, lang)

        X, y, groups, keys = load_dataset(folder)

        X_all.append(X)
        y_all.append(y)
        groups_all.append(groups)
        keys_all.extend(keys)

    return (
        np.vstack(X_all),
        np.concatenate(y_all),
        np.concatenate(groups_all),
        keys_all
    )


# ############################################################
# Main
############################################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True
    )

    parser.add_argument(
        "--model_out",
        default="model.pkl"
    )

    args = parser.parse_args()

    print("Loading dataset...")

    X, y, groups, keys = load_all_datasets(args.data_dir)

    print(f"Samples : {len(y)}")
    print(f"Features: {X.shape[1]}")
    print("=" * 50)
    print("Feature Statistics")
    print("=" * 50)

    print("NaN :", np.isnan(X).sum())
    print("Inf :", np.isinf(X).sum())
    print("Max :", np.max(X))
    print("Min :", np.min(X))

    # Replace problematic values if any
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6
    )

    ########################################################
    # Models
    #######################################################

    models = {

    "Logistic Regression":

        Pipeline([

            ("scaler", StandardScaler()),

            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            ))

        ]),

    "Random Forest":

        RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ),

    "Extra Trees":

        ExtraTreesClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ),

    "HistGradientBoosting":

        HistGradientBoostingClassifier(
            random_state=42
        )
}

    ########################################################
    # Cross Validation
    ########################################################

    cv = GroupKFold(n_splits=5)

    best_name = None
    best_model = None
    best_auc = -1

    for name, model in models.items():

        print()

        print("=" * 50)
        print(name)
        print("=" * 50)

        acc_scores = []
        auc_scores = []

        for train_idx, test_idx in cv.split(
                X,
                y,
                groups
        ):

            model.fit(
                X[train_idx],
                y[train_idx]
            )

            pred = model.predict(
                X[test_idx]
            )

            prob = model.predict_proba(
                X[test_idx]
            )[:, 1]

            acc = accuracy_score(
                y[test_idx],
                pred
            )

            auc = roc_auc_score(
                y[test_idx],
                prob
            )

            acc_scores.append(acc)
            auc_scores.append(auc)

        print(
            f"Accuracy : {np.mean(acc_scores):.4f}"
        )

        print(
            f"AUC      : {np.mean(auc_scores):.4f}"
        )

        if np.mean(auc_scores) > best_auc:

            best_auc = np.mean(auc_scores)

            best_model = model

            best_name = name

    ########################################################
    # Train on Full Dataset
    ########################################################

    print()
    print("=" * 50)
    print("Best Model")
    print("=" * 50)

    print(best_name)
 
    best_model.fit(X, y)

    joblib.dump(
        best_model,
        args.model_out
    )

    print()
    print(f"Saved model -> {args.model_out}")
    tree_model = best_model

    # Unwrap pipeline if necessary
    if hasattr(best_model, "named_steps"):
        tree_model = best_model.named_steps.get("clf", best_model)

    if hasattr(tree_model, "feature_importances_"):

        importance = tree_model.feature_importances_

        idx = np.argsort(importance)[::-1]

        print("\nTop 20 Features")

        for i in idx[:20]:
            print(f"{i:3d}: {importance[i]:.6f}")


if __name__ == "__main__":
    main()
