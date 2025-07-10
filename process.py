#!/usr/bin/env python3
import os
from PIL import Image
from torchvision import transforms

# --- CONFIG ---
INPUT_DIR  = "unprocessed-unhealthy"
OUTPUT_DIR = "processed-unhealthy"
TARGET_SIZE = 512  # final size (square)

# make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# define your augmentation + preprocessing pipeline
pipeline = transforms.Compose([
    # random zoom & crop
    transforms.RandomResizedCrop(
        TARGET_SIZE,
        scale=(0.8, 1.0),    # zoom between 80%–100% of original area
        ratio=(0.9, 1.1)     # allow slight aspect‐ratio shifts
    ),
    # random flips
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    # small random rotation
    transforms.RandomRotation(degrees=30),
    # small translation + additional zoom
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),   # up to 10% shift
        scale=(0.9, 1.1)        # up to ±10% scaling
    ),
    # photometric transforms
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15,
        hue=0.05
    ),
    # ensure final resize (in case RandomAffine or others changed size)
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
])

# gather all image files
files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
])

for idx, fname in enumerate(files):
    # open & convert to RGB
    img_path = os.path.join(INPUT_DIR, fname)
    img = Image.open(img_path).convert("RGB")

    # apply augmentation + preprocessing
    img_proc = pipeline(img)

    # build new filename, zero-padded 3 digits
    new_name = f"U-{idx:03d}.png"
    out_path = os.path.join(OUTPUT_DIR, new_name)

    # save as PNG
    img_proc.save(out_path, format="PNG")

    print(f"[{idx+1:03d}/{len(files):03d}] → {new_name}")

print("Done! All images are processed and saved to", OUTPUT_DIR)
