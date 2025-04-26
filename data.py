import os
from collections import Counter

from datasets import load_dataset
import numpy as np
from tqdm import tqdm


# ----------------------------------------------------------------------------
# ---- Create "data" Directory for Images and Labels ----

# Create "data" directory to hold images and labels
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)

# Create "images" directory to hold image data
images_dir = os.path.join(data_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# Create "labels" directory to hold image labels
labels_dir = os.path.join(data_dir, "labels")
os.makedirs(labels_dir, exist_ok=True)

# ----------------------------------------------------------------------------
# ---- Function for Loading Dataset Split ----


def load_split(split):
    # https://huggingface.co/datasets/ylecun/mnist
    ds = load_dataset("ylecun/mnist", split=split,)
    ds = ds.shuffle(seed=1337)

    print(f"Counts for each class in {split} split: "
          f"{list(Counter(ds['label']).values())}")

    return ds

# ----------------------------------------------------------------------------
# ---- Load Dataset Splits and Save Images and Labels to File as Binary ----


def main():
    for split in ["train", "test"]:
        ds = load_split(split)

        split_length = len(ds)
        progress_bar = None
        if progress_bar is None:
            progress_bar = tqdm(total=split_length,
                                unit=" Examples",
                                desc=f"{split} split")


        with open(f"data/images/{split}_images.bin", "wb") as img_file:
            for img in ds["image"]:
                img_arr = np.array(img)
                img_arr.astype(np.uint8).tofile(img_file)
                progress_bar.update(1)

        with open(f"data/labels/{split}_labels.bin", "wb") as labels_file:
            for label in ds["label"]:
                label_arr = np.array(label)
                label_arr.astype(np.uint8).tofile(labels_file)

        progress_bar = None


if __name__ == '__main__':
    main()