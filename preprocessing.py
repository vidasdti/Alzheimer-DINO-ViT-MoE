import os
import random
from PIL import Image

from configs import (
    RAW_DATASET,
    PROCESSED_DATASET,
    CLASS_FOLDER_MAP,
    IMAGE_SIZE,
    TRAIN_RATIO,
    RANDOM_SEED,
)

random.seed(RANDOM_SEED)

LOG_FILE = os.path.join(PROCESSED_DATASET, "processing_log.txt")

# ==========================================================
# LOG
# ==========================================================

def write_log(text):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# ==========================================================
# PAD + RESIZE
# ==========================================================

def pad_to_center(image, size=IMAGE_SIZE):

    image.thumbnail(size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", size, (0, 0, 0))

    left = (size[0] - image.width) // 2
    top = (size[1] - image.height) // 2

    canvas.paste(image, (left, top))

    return canvas

# ==========================================================
# PROCESS CLASS
# ==========================================================

def process_class(original_name, output_name):

    print(f"\nProcessing {original_name} -> {output_name}")

    input_dir = os.path.join(RAW_DATASET, original_name)

    if not os.path.exists(input_dir):
        write_log(f"[MISSING] {input_dir}")
        return

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])

    random.shuffle(files)

    n_train = int(len(files) * TRAIN_RATIO)

    train_files = files[:n_train]
    test_files = files[n_train:]

    train_dir = os.path.join(PROCESSED_DATASET, "train",output_name)
    test_dir = os.path.join(PROCESSED_DATASET, "test", output_name)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # -------------------------
    # Train
    # -------------------------

    counter = 0

    for file in train_files:
        src = os.path.join(input_dir, file)
        try:
            with Image.open(src) as image:
                img = image.convert("RGB")

        except Exception as e:
            write_log(f"[CORRUPT] {src} | {e}")
            continue

        img = pad_to_center(img)
        save_name = f"image{counter}.png"
        save_path = os.path.join(train_dir, save_name)
        img.save(save_path)

        counter += 1

    # -------------------------
    # Test
    # -------------------------

    counter = 0

    for file in test_files:

        src = os.path.join(input_dir, file)

        try:
            with Image.open(src) as image:
                img = image.convert("RGB")

        except Exception as e:
            write_log(f"[CORRUPT] {src} | {e}")
            continue

        img = pad_to_center(img)
        save_name = f"image{counter}.png"
        save_path = os.path.join(test_dir, save_name)
        img.save(save_path)

        counter += 1

    print(f"Total : {len(files)}")
    print(f"Train : {len(train_files)}")
    print(f"Test  : {len(test_files)}")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    os.makedirs(PROCESSED_DATASET, exist_ok=True)

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    for original_name, output_name in sorted(CLASS_FOLDER_MAP.items()):
        process_class(original_name, output_name)

    print("\n====================================")
    print("Dataset preparation completed.")
    print(f"Output : {PROCESSED_DATASET}")
    print(f"Log    : {LOG_FILE}")
    print("====================================")