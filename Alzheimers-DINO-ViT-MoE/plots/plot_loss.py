import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs import OUTPUT_DIR, PLOT_DIR, NUM_FOLDS

# ============================================
# Settings
# ============================================

LOG_DIR = os.path.join(OUTPUT_DIR, "logsDINO")
SAVE_PATH = os.path.join(PLOT_DIR, "LossCurve.png")

os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================
# Load logs
# ============================================

train_mix_losses = []
train_clean_losses = []
val_losses = []

max_epochs = 0

for fold in range(1, NUM_FOLDS + 1):

    csv_path = os.path.join(LOG_DIR, f"epoch_logs_fold{fold}.csv")

    if not os.path.exists(csv_path):
        print(f"[WARNING] Missing file: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # required columns (based on your training code)
    train_mix_losses.append(df["TrainLoss_Mix"].values)
    train_clean_losses.append(df["TrainLoss_Clean"].values)
    val_losses.append(df["ValLoss"].values)

    max_epochs = max(max_epochs, len(df))

# ============================================
# Safety check
# ============================================

if len(train_mix_losses) == 0:
    raise RuntimeError("No logs found. Check LOG_DIR path.")

# ============================================
# Padding (handles early stopping correctly)
# ============================================

def pad(arr, target_len):
    arr = np.array(arr)
    if len(arr) < target_len:
        pad_value = arr[-1]
        arr = np.concatenate([arr, np.repeat(pad_value, target_len - len(arr))])
    return arr

train_mix_losses = np.array([pad(x, max_epochs) for x in train_mix_losses])
train_clean_losses = np.array([pad(x, max_epochs) for x in train_clean_losses])
val_losses = np.array([pad(x, max_epochs) for x in val_losses])

epochs = np.arange(1, max_epochs + 1)

# ============================================
# Mean ± Std
# ============================================

mix_mean = train_mix_losses.mean(axis=0)
mix_std = train_mix_losses.std(axis=0, ddof=1)

clean_mean = train_clean_losses.mean(axis=0)
clean_std = train_clean_losses.std(axis=0, ddof=1)

val_mean = val_losses.mean(axis=0)
val_std = val_losses.std(axis=0, ddof=1)

# ============================================
# Plot
# ============================================

plt.figure(figsize=(9, 6))

# Train Mixup
plt.plot(epochs, mix_mean, linewidth=2.2, label="Train Loss (with Mixup)", color="blue")
plt.fill_between(epochs, mix_mean - mix_std, mix_mean + mix_std, alpha=0.15, color="blue")

# Train Clean
plt.plot(epochs, clean_mean, linewidth=2.2, label="Train Loss (without Mixup)", color="green")
plt.fill_between(epochs, clean_mean - clean_std, clean_mean + clean_std, alpha=0.15, color="green")

# Validation
plt.plot(epochs, val_mean, linewidth=2.2, label="Validation Loss", color="orange")
plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.15, color="orange")

# ============================================
# Styling
# ============================================

plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.title("Training vs Validation Loss (5-Fold Mean ± Std)", fontsize=14)

plt.xlim(1, max_epochs)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {SAVE_PATH}")