import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

from PIL import Image
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs import *
from models import DINOWrapper, DINO_ViT_MoE
from dataset import val_transform
# ======================================
# Config
# ======================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================
# Load Model
# ======================================

def load_model(fold):

    dino = DINOWrapper(
        model_name="vit_base_patch16_224_dino",
        pretrained=False,
        device=DEVICE
    )

    model = DINO_ViT_MoE(dino, NUM_CLASSES, num_experts=NUM_EXPERTS,
        finetune_vit_layers=FINETUNE_VIT_LAYERS)

    model.load_state_dict( torch.load(os.path.join(OUTPUT_DIR, 
        f"best_dinovit_moe_fold{fold}.pth"), map_location=DEVICE, weights_only=True))

    model.to(DEVICE)
    model.eval()

    return model


# ======================================
# Validation Images
# ======================================

def load_validation_list(fold):

    txt = os.path.join(OUTPUT_DIR, "val_listsDINO", f"val_fold{fold}.txt")

    with open(txt) as f:
        paths = [x.strip() for x in f.readlines()]

    labels = []

    for p in paths:

        cls = os.path.basename(os.path.dirname(p))
        labels.append(LABEL_MAP[cls])

    return paths, labels


# ======================================
# Inference
# ======================================

@torch.no_grad()
def predict(model, paths):

    probs = []

    for path in paths:

        img = Image.open(path).convert("RGB")

        img = val_transform(img).unsqueeze(0).to(DEVICE)

        logits = model(img)

        prob = torch.softmax(
            logits,
            dim=1
        ).cpu().numpy()[0]

        probs.append(prob)

    return np.array(probs)


# ======================================
# Collect all folds
# ======================================

all_probs = []
all_labels = []

for fold in range(1, NUM_FOLDS+1):

    print(f"Fold {fold}")

    probs = np.load(os.path.join(OUTPUT_DIR, f"y_prob_fold{fold}.npy"))
    labels = np.load(os.path.join(OUTPUT_DIR, f"y_true_fold{fold}.npy"))

    all_probs.append(probs)
    all_labels.append(np.array(labels))

all_probs = np.vstack(all_probs)
all_labels = np.concatenate(all_labels)

# ======================================
# ROC
# ======================================

y_true = label_binarize(all_labels, classes=np.arange(NUM_CLASSES))
plt.figure(figsize=(7,7))

for i, cls in enumerate(CLASS_NAMES):

    fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{cls} (AUC={roc_auc:.3f})")

plt.plot([0,1], [0,1], "--", color="black")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

SAVE_PATH = os.path.join(PLOT_DIR, "roc_curve.png")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.savefig(SAVE_PATH, dpi=300)
plt.close()

print(SAVE_PATH)