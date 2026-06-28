import os
import numpy as np
from torch.utils.data import DataLoader,WeightedRandomSampler

from dataset import AlzheimerDataset, train_transform, val_transform, test_transform
from configs import BATCH_SIZE, NUM_CLASSES

def collect_dataset(root_dir, class_names, label_map):
    """
    Collect image paths and labels from dataset folders.
    """
    paths = []
    labels = []

    for cls in class_names:

        # try both structures:
        possible_folders = [
            os.path.join(root_dir, "train", cls),
            os.path.join(root_dir, cls)]

        folder = None
        for f in possible_folders:
            if os.path.exists(f):
                folder = f
                break

        if folder is None:
            continue

        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(folder, fname))
                labels.append(label_map[cls])

    return np.array(paths), np.array(labels)


def build_dataloaders(train_files, train_labels, val_files, val_labels, num_workers):
    """
    Collect image paths and labels from dataset folders.
    """

    train_ds = AlzheimerDataset(list(zip(train_files, train_labels)), train_transform)
    val_ds = AlzheimerDataset(list(zip(val_files, val_labels)), val_transform)

    labels_all = np.array([lbl for _, lbl in train_ds.samples], dtype=int)
    class_counts = np.bincount(labels_all, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-12)
    sample_weights = np.array([class_weights[int(lbl)] for lbl in labels_all], dtype=np.float32)

    sampler = WeightedRandomSampler(weights=sample_weights, 
        num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    return train_loader, val_loader

def build_test_loader(test_paths, test_labels, num_workers=2):

    test_ds = AlzheimerDataset(list(zip(test_paths, test_labels)), test_transform  )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return test_loader