import gc
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from torch_optimizer import Lookahead

from configs import *
from models import (DINOWrapper, DINO_ViT_MoE, LabelSmoothingCrossEntropy)
from utils import (ci_95, evaluate, evaluate_clean, save_confusion, summarize_folders, train_epoch)
from dataloader import (collect_dataset, build_dataloaders)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_output_dirs():
    folders = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "val_listsDINO"),
        os.path.join(OUTPUT_DIR, "confusionsDINO"),
        os.path.join(OUTPUT_DIR, "logsDINO"),
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device):
    dino_wrapper = DINOWrapper(
        model_name="vit_base_patch16_224_dino",
        pretrained=True,
        device=device,
    )

    dino_wrapper.unfreeze_last_n_blocks(DINO_UNFREEZE_LAST)

    model = DINO_ViT_MoE(
        dino_wrapper,
        NUM_CLASSES,
        num_experts=NUM_EXPERTS,
        finetune_vit_layers=FINETUNE_VIT_LAYERS,
    )

    return model.to(device), dino_wrapper

# ---------- main ----------
def main():

    set_seed(42)
    create_output_dirs()

    print(f" Starting {NUM_FOLDS}-Fold Cross Validation with DINO+ViT+MoE")
    print(" Loading DINO ViT (timm)...")

    train_root = os.path.join(PROCESSED_DATASET, "train")
    paths, labels = collect_dataset(train_root, CLASS_NAMES, LABEL_MAP)

    summarize_folders(PROCESSED_DATASET, CLASS_NAMES)

    if len(paths) == 0:
        raise RuntimeError(f"No images found in {PROCESSED_DATASET}. Check the dataset paths.")

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    # DataLoader workers safe default (reduce if Windows issues)
    num_workers = min(4, max(0, os.cpu_count() - 1))

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels), start=1):

        ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_fold{fold}.pth")
        start_epoch = 1
        print(f"\n========== Fold {fold}/{NUM_FOLDS} ==========")
        train_files, val_files = paths[train_idx], paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        print(f"  -> Train samples: {len(train_files)}, Val samples: {len(val_files)}")

        # save val list
        val_txt_path = os.path.join(OUTPUT_DIR, "val_listsDINO", f"val_fold{fold}.txt")
        with open(val_txt_path, "w", encoding="utf-8") as f:
            for vf in val_files:
                f.write(vf + "\n")
        print(f" Val file list saved: {val_txt_path}")

        train_loader, val_loader = build_dataloaders(train_files, train_labels,
            val_files, val_labels, num_workers)
        
        model, dino_wrapper = build_model(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        base_opt = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer = Lookahead(base_opt, k=5, alpha=0.5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, EPOCHS // 2), T_mult=1, eta_min=LR*0.1
        )

        criterion = LabelSmoothingCrossEntropy(0.05)
        scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

        start_epoch = 1
        best_acc = 0.0
        no_improve = 0
        epoch_logs = []

        ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_fold{fold}.pth")

        if os.path.exists(ckpt_path):
            print(f" Resuming Fold {fold}...")

            checkpoint = torch.load(ckpt_path, map_location=device)

            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])

            start_epoch = checkpoint["epoch"] + 1

        for epoch in range(start_epoch, EPOCHS + 1):
            t0 = time.time()
            print(f"\n Epoch {epoch}/{EPOCHS}")
            train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, scaler, device, GRAD_CLIP)
            elapsed = time.time() - t0
            print(f" Train Mixup -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro-F1: {train_f1:.4f} (time: {elapsed:.1f}s)")
            # ----- Compute CLEAN (true) train metrics WITHOUT mixup -----
            train_loss_clean, train_acc_clean, train_f1_clean = evaluate_clean(model, train_loader, criterion, device)
            print(f" Train Clean -> Loss: {train_loss_clean:.4f}, Acc: {train_acc_clean:.4f}, F1: {train_f1_clean:.4f}")

            y_true, y_pred, y_prob, val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            print(f" Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1:.4f}")
            print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

            epoch_logs.append([
            epoch, train_loss, train_acc, train_f1,            # mixup
            train_loss_clean, train_acc_clean, train_f1_clean,   # clean
            val_loss, val_acc, val_f1])


            # Save epoch log CSV
            log_txt_path = os.path.join(OUTPUT_DIR, "logsDINO", f"epoch_logs_fold{fold}.csv")
            df_logs = pd.DataFrame(epoch_logs, columns=LOG_COLUMNS)
            df_logs.to_csv(log_txt_path, index=False)

            # Save checkpoint (full checkpoint)
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": base_opt.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch
            }
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_fold{fold}.pth")
            torch.save(checkpoint, ckpt_path)

            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0

                np.save(os.path.join(OUTPUT_DIR, f"y_true_fold{fold}.npy"), y_true)
                np.save(os.path.join(OUTPUT_DIR, f"y_pred_fold{fold}.npy"), y_pred)
                np.save(os.path.join(OUTPUT_DIR, f"y_prob_fold{fold}.npy"), y_prob)

                best_model_path = os.path.join(OUTPUT_DIR, f"best_dinovit_moe_fold{fold}.pth")
                torch.save(model.state_dict(), best_model_path)
                cm_path = save_confusion(y_true, y_pred, fold, CLASS_NAMES, os.path.join(OUTPUT_DIR, "confusionsDINO"))
                print(f" Best model saved: {best_model_path}")
                print(f"   Confusion saved: {cm_path}")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(" Early stopping.")
                    break

            scheduler.step()

        # save CSV summary for this fold
        df_fold = pd.DataFrame(epoch_logs, columns=LOG_COLUMNS)

        summary_path = os.path.join(OUTPUT_DIR, f"summary_fold{fold}.csv")
        df_fold.to_csv(summary_path, index=False)
        fold_results.append(best_acc)
        # Free GPU memory before next fold
        del model
        del optimizer
        del scheduler
        del scaler
        del base_opt
        del dino_wrapper

        torch.cuda.empty_cache()
        gc.collect()

    # final summary
    print("\n Cross Validation Results")
    for i, acc in enumerate(fold_results, 1):
        print(f"Fold {i}: Best Val Acc = {acc:.4f}")
    mean_acc, ci = ci_95(fold_results)
    print(f"Mean Acc: {mean_acc:.4f}, Std: {np.std(fold_results):.4f}, 95% CI: ±{ci:.4f}")

if __name__ == "__main__":
    main()
