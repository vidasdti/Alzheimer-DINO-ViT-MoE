# ======================================
# Training
# ======================================

NUM_CLASSES = 4
EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = 224

LR = 1e-4
WEIGHT_DECAY = 0.05

PATIENCE = 5
NUM_FOLDS = 5

NUM_EXPERTS = 3
FINETUNE_VIT_LAYERS = 3
DINO_UNFREEZE_LAST = 8

GRAD_CLIP = 1.0

# ======================================
# DINO
# ======================================

DINO_MEAN = [0.485,0.456,0.406]
DINO_STD = [0.229,0.224,0.225]

# ======================================
# Dataset
# ======================================

CLASS_NAMES = ["MD","MOD","ND","VMD"]

LABEL_MAP = {
    "MD": 0,
    "MOD": 1,
    "ND": 2,
    "VMD": 3
}

# ======================================
# Preprocessing
# ======================================

RAW_DATASET = "raw_dataset"
PROCESSED_DATASET = "input"
OUTPUT_DIR = "output" 
PLOT_DIR = "images"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

IMAGE_SIZE = (IMG_SIZE, IMG_SIZE)

CLASS_FOLDER_MAP = {
    "MildDemented": "MD",
    "ModerateDemented": "MOD",
    "NonDemented": "ND",
    "VeryMildDemented": "VMD"
}

LOG_COLUMNS = ["Epoch", "TrainLoss_Mix", "TrainAcc_Mix", "TrainF1_Mix",
    "TrainLoss_Clean", "TrainAcc_Clean",  "TrainF1_Clean",
    "ValLoss", "ValAcc", "ValF1"]