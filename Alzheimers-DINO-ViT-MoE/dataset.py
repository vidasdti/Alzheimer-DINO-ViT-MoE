import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from configs import DINO_MEAN, DINO_STD, IMG_SIZE

# ---------- Transforms ----------
class EnsureThreeChannels:
    def __call__(self, x):
        # x is tensor with shape (C,H,W)
        return x.repeat(3,1,1) if x.shape[0] == 1 else x

normalize = transforms.Normalize(mean=DINO_MEAN, std=DINO_STD)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(12),
    transforms.RandomAffine(degrees=0,translate=(0.02, 0.02),scale=(0.97, 1.03)),
    transforms.RandomResizedCrop(IMG_SIZE,scale=(0.95, 1.00)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3,sigma=(0.1, 0.4))], p=0.4),
    transforms.ToTensor(),
    EnsureThreeChannels(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    EnsureThreeChannels(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    EnsureThreeChannels(),
    normalize,
])

# ---------- Dataset ----------
class AlzheimerDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(int(label), dtype=torch.long)

        return image, label
    
__all__ = [
    "AlzheimerDataset",
    "train_transform",
    "val_transform",
]