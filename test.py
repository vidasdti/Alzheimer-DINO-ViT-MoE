import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from dataloader import collect_dataset, build_test_loader
from models import DINOWrapper, DINO_ViT_MoE, LabelSmoothingCrossEntropy
from configs import *
from utils import evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Model
# -------------------------
def build_model(device):
    dino_wrapper = DINOWrapper(
        model_name="vit_base_patch16_224_dino",
        pretrained=False,
        device=device,
    )

    dino_wrapper.unfreeze_last_n_blocks(DINO_UNFREEZE_LAST)

    model = DINO_ViT_MoE(
        dino_wrapper,
        NUM_CLASSES,
        num_experts=NUM_EXPERTS,
        finetune_vit_layers=FINETUNE_VIT_LAYERS,
    )

    return model.to(device)


# -------------------------
# Plot Confusion Matrix
# -------------------------
def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------
# TEST
# -------------------------
def run_test():

    print("\n FINAL TEST ON SEPARATE HELD-OUT TEST SET")

    test_root = os.path.join(PROCESSED_DATASET, "test")

    test_paths, test_labels = collect_dataset(
        test_root,
        CLASS_NAMES,
        LABEL_MAP
    )

    print(f"Test samples: {len(test_paths)}")

    test_loader = build_test_loader(
        test_paths,
        test_labels,
        num_workers=2
    )

    criterion = LabelSmoothingCrossEntropy(0.05)

    test_out_dir = os.path.join(OUTPUT_DIR, "test_results")
    cm_out_dir = os.path.join(test_out_dir, "confusion_matrices")
    os.makedirs(cm_out_dir, exist_ok=True)

    all_fold_results = []

    for fold in range(1, NUM_FOLDS + 1):

        print(f"\n========== Fold {fold} TEST ==========")

        model = build_model(device)

        ckpt_path = os.path.join(
            OUTPUT_DIR,
            f"best_dinovit_moe_fold{fold}.pth"
        )

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state)

        model.eval()

        with torch.no_grad():
            y_true, y_pred, y_prob, loss, acc, f1 = evaluate(
                model,
                test_loader,
                criterion,
                device
            )

        # -------------------------
        # Metrics
        # -------------------------
        print(f"Loss: {loss:.4f}")
        print(f"Acc : {acc:.4f}")
        print(f"F1  : {f1:.4f}")

        print(
            classification_report(
                y_true,
                y_pred,
                target_names=CLASS_NAMES,
                digits=4
            )
        )

        # -------------------------
        # Confusion Matrix
        # -------------------------
        cm = confusion_matrix(y_true, y_pred)

        save_path = os.path.join(
            test_out_dir,
            f"cm_fold{fold}.png"
        )

        plot_confusion_matrix(
            cm,
            CLASS_NAMES,
            save_path,
            title=f"Confusion Matrix - Fold {fold}"
        )

        print(f" Confusion matrix saved: {save_path}")
        print("-" * 60)
        
        all_fold_results.append([acc, f1])

        del model
        torch.cuda.empty_cache()

    # -------------------------
    # Final Summary
    # -------------------------
    all_fold_results = np.array(all_fold_results)

    print("\n📊 FINAL TEST SUMMARY")
    print(f"Accuracy : {all_fold_results[:,0].mean():.4f} ± {all_fold_results[:,0].std():.4f}")
    print( f"Macro F1 : {all_fold_results[:,1].mean():.4f} ± {all_fold_results[:,1].std():.4f}")


if __name__ == "__main__":
    run_test()
