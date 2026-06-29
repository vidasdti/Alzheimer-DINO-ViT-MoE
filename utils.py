import os
import numpy as np
import scipy.stats as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ============================================================
# Visualization
# ============================================================
def save_confusion(y_true, y_pred, fold, class_names, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Fold {fold}")
    save_path = os.path.join(out_dir, f"confusion_fold{fold}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path

# ============================================================
# Summary of folders
# ============================================================
def summarize_folders(root_train, class_names):
    print(" Summary of folders and file counts:")

    summary = {}
    total = 0

    for cls in class_names:
        folder = os.path.join(root_train, "train", cls)
        cnt = 0
        if os.path.exists(folder):
            cnt = sum(
                1 for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
        summary[cls] = {"train": cnt}
        total += cnt
        print(f"  - {cls}: train={cnt}")

    print(f" Total train files: {total}")

    return summary

# ============================================================
# Statistics
# ============================================================
def ci_95(acc_list):
    """
    Returns mean and 95% confidence interval.
    """
    if len(acc_list) == 0:
        return 0.0, 0.0
    
    mean = float(np.mean(acc_list))

    if len(acc_list) > 1:
        sem = float(st.sem(acc_list))
        ci = sem * st.t.ppf((1+0.95)/2., len(acc_list)-1)
    else:
        ci = 0.0

    return mean, ci


# ============================================================
# Mixup
# ============================================================
def mixup_data(x, y, alpha_range=(0.05, 0.2)):
    alpha = np.random.uniform(*alpha_range)
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = (y, y[index], lam)
    return mixed_x, mixed_y

def mixup_criterion(criterion, pred, mixed_y):
    y_a, y_b, lam = mixed_y
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# Training
# ============================================================
def train_epoch(model, loader, optimizer, criterion, scaler, device, grad_clip=None,):
    
    model.train()
    total_loss, total = 0.0, 0
    correct_weighted = 0.0
    y_true_for_f1, y_pred_for_f1 = [], []

    for x, y in tqdm(loader, desc=" Training", ncols=120):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        mixed_x, mixed_y = mixup_data(x, y)
        y_a, y_b, lam = mixed_y

        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(mixed_x)  # predict on mixed inputs
            loss = mixup_criterion(criterion, logits, mixed_y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        # ----- Mixup-aware accuracy computation -----
        preds = logits.argmax(dim=1)  # (B,)
        # Ensure lambda is a tensor on the correct device
        if not isinstance(lam, torch.Tensor):
            lam_t = torch.tensor(lam, device=y.device, dtype=torch.float32)
        else:
            lam_t = lam.to(y.device).float()
        # Handle both scalar and tensor lambda values
        if lam_t.ndim == 0:
            lam_batch = lam_t
        else:
            lam_batch = lam_t

        # Weighted accuracy:
        # lam * correct(y_a) + (1-lam) * correct(y_b)
        correct_a = (preds == y_a).float()
        correct_b = (preds == y_b).float()
        
        weighted_correct = (lam_batch * correct_a + (1.0 - lam_batch) * correct_b).sum().item()
        correct_weighted += weighted_correct

        # Store predictions for an approximate macro-F1 estimate.
        # Note: F1 during Mixup training is only an approximation.
        # Validation F1 should be used as the primary metric.        
        y_true_for_f1.extend(y.cpu().tolist())
        y_pred_for_f1.extend(preds.cpu().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct_weighted / total if total > 0 else 0.0
    # macro f1 on the approximated lists (note: this is only an approximation when mixup used)
    try:
        macro_f1 = f1_score(y_true_for_f1, y_pred_for_f1, average="macro")
    except Exception:
        macro_f1 = 0.0
    return avg_loss, acc, macro_f1

# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []
    y_prob = []

    with torch.amp.autocast(
        device_type="cuda",
        enabled=torch.cuda.is_available()
    ):

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(logits, y)

            batch = y.size(0)

            total_loss += loss.item() * batch
            total += batch

            probs = torch.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)

            correct += (preds == y).sum().item()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    val_loss = total_loss / total
    val_acc = correct / total
    val_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob),
        val_loss,
        val_acc,
        val_f1,
    )

@torch.no_grad()
def evaluate_clean(model, loader, criterion, device):
    """
    Evaluate on train set WITHOUT mixup to get true metrics.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred = [], []

    for x, y in tqdm(loader, desc="Evaluating", ncols=120):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        batch = y.size(0)
        total_loss += loss.item() * batch
        total += batch

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()

        y_true += y.cpu().tolist()
        y_pred += preds.cpu().tolist()

    avg_loss = total_loss / total
    acc = correct / total
    f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, acc, f1


__all__ = [
    "save_confusion",
    "ci_95",
    "mixup_data",
    "mixup_criterion",
    "train_epoch",
    "evaluate",
]

