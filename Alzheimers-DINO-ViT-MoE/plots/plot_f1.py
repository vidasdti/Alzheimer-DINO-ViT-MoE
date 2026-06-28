import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs import OUTPUT_DIR, PLOT_DIR

# ======================================
# Config
# ======================================

LOG_DIR = os.path.join(OUTPUT_DIR, "logsDINO")
SAVE_PATH = os.path.join(PLOT_DIR, "F1Curve.png")

os.makedirs(PLOT_DIR, exist_ok=True)

# ======================================
# Load logs
# ======================================

csv_files = sorted(glob.glob(os.path.join(LOG_DIR, "epoch_logs_fold*.csv")))

train_mix_f1s = []
train_clean_f1s = []
val_f1s = []

max_epoch = 0

for file in csv_files:

    df = pd.read_csv(file)

    # required columns from your training loop:
    train_mix_f1s.append(df["TrainF1_Mix"].values)
    train_clean_f1s.append(df["TrainF1_Clean"].values)
    val_f1s.append(df["ValF1"].values)

    max_epoch = max(max_epoch, len(df))

# ======================================
# Pad (handles early stopping)
# ======================================

def pad(arr):
    arr = np.array(arr)
    if len(arr) < max_epoch:
        arr = np.concatenate([arr, np.repeat(arr[-1], max_epoch - len(arr))])
    return arr

train_mix_f1s = np.array([pad(x) for x in train_mix_f1s])
train_clean_f1s = np.array([pad(x) for x in train_clean_f1s])
val_f1s = np.array([pad(x) for x in val_f1s])

epochs = np.arange(1, max_epoch + 1)

# ======================================
# Mean ± STD
# ======================================

mix_mean = train_mix_f1s.mean(axis=0)
mix_std = train_mix_f1s.std(axis=0, ddof=1)

clean_mean = train_clean_f1s.mean(axis=0)
clean_std = train_clean_f1s.std(axis=0, ddof=1)

val_mean = val_f1s.mean(axis=0)
val_std = val_f1s.std(axis=0, ddof=1)

# ======================================
# Plot
# ======================================

plt.figure(figsize=(9, 6))

# Train Mixup F1
plt.plot(epochs, mix_mean, linewidth=2.5, label="Train Macro-F1 (with Mixup)", color="blue")
plt.fill_between(epochs, mix_mean - mix_std, mix_mean + mix_std, alpha=0.15, color="blue")

# Train Clean F1
plt.plot(epochs, clean_mean, linewidth=2.5, label="Train Macro-F1 (without Mixup)", color="green")
plt.fill_between(epochs, clean_mean - clean_std, clean_mean + clean_std, alpha=0.15, color="green")

# Validation F1
plt.plot(epochs, val_mean, linewidth=2.5, label="Validation Macro-F1", color="orange")
plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.15, color="orange")

# ======================================
# Styling
# ======================================

plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Macro F1 Score", fontsize=13)
plt.title("Macro-F1 Curve (5-Fold Mean ± Std)", fontsize=14)

plt.xlim(1, max_epoch)
plt.ylim(0, 1)

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {SAVE_PATH}")