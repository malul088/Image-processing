import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ── Configuration ─────────────────────────────────────────────────────────────
FEATURE_INDICES = [1, 4]   # mean texture, mean smoothness — change freely
LEARNING_RATE   = 0.1
EPOCHS          = 200
RANDOM_SEED     = 42

def main():

    # ── 1. Load and prepare data ──────────────────────────────────────────────
    data         = load_breast_cancer()
    X_all        = data.data
    y            = data.target.astype(float)
    feature_names = data.feature_names

    X = X_all[:, FEATURE_INDICES]
    selected_names = [feature_names[i] for i in FEATURE_INDICES]
    n_features = X.shape[1]

    print(f"Features selected : {selected_names}")
    print(f"Dataset shape     : {X.shape}  (samples × features)")
    print(f"Class distribution: {int(y.sum())} benign, {int((1-y).sum())} malignant")

    print("=== Before Normalization ===")
    means_before = np.mean(X, axis=0)
    stds_before = np.std(X, axis=0)
    for i, name in enumerate(selected_names):
        print(f"Feature: {name:<20} | Mean: {means_before[i]:.4f} | Std: {stds_before[i]:.4f}")
    print("-" * 60)

    # ── 2. Normalise ──────────────────────────────────────────────────────────
    # Gradient descent is sensitive to feature scale.
    # StandardScaler gives each feature mean=0, std=1.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


if __name__ == '__main__':
    main()