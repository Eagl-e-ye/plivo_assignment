"""
Train a CNN on MFCC spectrograms for End-of-Turn prediction.
"""

import argparse
import copy

import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

############################################################
# Dataset
############################################################

class MFCCDataset(Dataset):
    """
    Dataset for MFCC spectrograms.

    X shape:
        (N,1,40,150)
    """

    def __init__(self, X, y, train=False):

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        x = self.X[idx].copy()

        if self.train:
            x = self.spec_augment(x)

        return (
            torch.from_numpy(x),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

    ########################################################
    # SpecAugment
    ########################################################

    def spec_augment(self, x):

        # x shape = (1,40,150)

        x = x.copy()

        ####################################################
        # Frequency masking
        ####################################################

        if np.random.rand() < 0.5:

            width = np.random.randint(2, 6)

            start = np.random.randint(0, 40 - width)

            x[:, start:start + width, :] = 0

        ####################################################
        # Time masking
        ####################################################

        if np.random.rand() < 0.5:

            width = np.random.randint(5, 20)

            start = np.random.randint(0, 150 - width)

            x[:, :, start:start + width] = 0

        ####################################################
        # Small Gaussian noise
        ####################################################

        if np.random.rand() < 0.5:

            noise = np.random.normal(
                0,
                0.01,
                x.shape
            )

            x += noise.astype(np.float32)

        return x


############################################################
# CNN
############################################################

class CNNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            ###############################################
            # Block 1
            ###############################################

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            ###############################################
            # Block 2
            ###############################################

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            ###############################################
            # Block 3
            ###############################################

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
# DataLoader
############################################################

def build_dataloaders(
    X,
    y,
    groups,
    batch_size=32
):

    splitter = GroupShuffleSplit(
        test_size=0.2,
        random_state=42,
        n_splits=1
    )

    train_idx, val_idx = next(

        splitter.split(
            X,
            y,
            groups
        )

    )

    train_dataset = MFCCDataset(
        X[train_idx],
        y[train_idx],
        train=True
    )

    val_dataset = MFCCDataset(
        X[val_idx],
        y[val_idx],
        train=False
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0

    )

    return (

        train_loader,

        val_loader,

        train_idx,

        val_idx

    )


############################################################
# Training Utilities
############################################################

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * X.size(0)

    return running_loss / len(loader.dataset)


############################################################
# Validation
############################################################

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_prob = []
    all_pred = []
    all_true = []

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = criterion(logits, y)

        running_loss += loss.item() * X.size(0)

        prob = torch.sigmoid(logits)

        pred = (prob >= 0.5).float()

        all_prob.extend(
            prob.cpu().numpy()
        )

        all_pred.extend(
            pred.cpu().numpy()
        )

        all_true.extend(
            y.cpu().numpy()
        )

    loss = running_loss / len(loader.dataset)

    acc = accuracy_score(
        all_true,
        all_pred
    )

    auc = roc_auc_score(
        all_true,
        all_prob
    )

    return loss, acc, auc


############################################################
# Main Training Function
############################################################

def train_model(args):

    ########################################################
    # Device
    ########################################################

    device = torch.device(

        "mps"

        if torch.backends.mps.is_available()

        else "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print("Device :", device)

    ########################################################
    # Load Dataset
    ########################################################

    data = np.load(args.dataset)

    X = data["X"]

    y = data["y"]

    groups = data["groups"]

    print("Dataset:", X.shape)

    ########################################################
    # DataLoaders
    ########################################################

    train_loader, val_loader, _, _ = build_dataloaders(

        X,
        y,
        groups,
        batch_size=args.batch_size

    )

    ########################################################
    # Model
    ########################################################

    model = CNNModel().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=args.lr,

        weight_decay=1e-4

    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3

    )

    ########################################################
    # Early Stopping
    ########################################################

    best_auc = -1

    best_epoch = 0

    best_weights = copy.deepcopy(
        model.state_dict()
    )

    patience_counter = 0

    ########################################################
    # Epoch Loop
    ########################################################

    for epoch in range(args.epochs):

        train_loss = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            device

        )

        val_loss, val_acc, val_auc = evaluate(

            model,

            val_loader,

            criterion,

            device

        )

        scheduler.step(val_auc)

        print(

            f"Epoch {epoch+1:02d} | "

            f"Train Loss {train_loss:.4f} | "

            f"Val Loss {val_loss:.4f} | "

            f"Acc {val_acc:.4f} | "

            f"AUC {val_auc:.4f}"

        )

        ###############################################
        # Save Best Model
        ###############################################

        if val_auc > best_auc:

            best_auc = val_auc

            best_epoch = epoch + 1

            patience_counter = 0

            best_weights = copy.deepcopy(

                model.state_dict()

            )

            torch.save(

                best_weights,

                args.output

            )

            print(

                "Saved Best Model"

            )

        else:

            patience_counter += 1

        ###############################################
        # Early Stop
        ###############################################

        if patience_counter >= args.patience:

            print()

            print("Early stopping")

            break

    ########################################################
    # Final Report
    ########################################################

    print()

    print("=" * 50)

    print("Training Finished")

    print("=" * 50)

    print("Best Epoch :", best_epoch)

    print("Best AUC   :", round(best_auc, 4))


############################################################
# Main
############################################################

def main():

    parser = argparse.ArgumentParser(
        description="Train CNN for End-of-Turn Prediction"
    )

    parser.add_argument(
        "--dataset",
        default="mfcc_dataset.npz",
        help="Path to MFCC dataset"
    )

    parser.add_argument(
        "--output",
        default="cnn_model.pt",
        help="Output model file"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CNN Training")
    print("=" * 60)

    train_model(args)


if __name__ == "__main__":
    main()