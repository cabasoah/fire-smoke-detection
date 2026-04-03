# A script (pipeline) for augmenting the train dataset.

import os
import cv2
import albumentations as A
import random

# --- Configuration ---
BASE_DIR = 'C:/Users/charl/Downloads/Ghana-fire/train'  
INSPECTION_DIR = 'C:/Users/charl/Downloads/Ghana-fire/inspection' # New folder for augmented images
TARGET_COUNT = 1000
CATEGORIES = ['fire', 'smoke']
IMG_SIZE = (224, 224) 

# Define the augmentation pipeline
augmenter = A.Compose([
    A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1], always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
])

def augment_to_target(category):
    path = os.path.join(BASE_DIR, category)
    # Create category subfolder in inspection directory
    inspect_path = os.path.join(INSPECTION_DIR, category)
    os.makedirs(inspect_path, exist_ok=True)
    
    images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    current_count = len(images)
    needed = TARGET_COUNT - current_count
    
    print(f"\n--- Processing {category.upper()} ---")
    
    print(f"Standardizing {current_count} original images to {IMG_SIZE}...")
    for img_name in images:
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        if image is not None:
            resized = cv2.resize(image, (IMG_SIZE[1], IMG_SIZE[0]))
            cv2.imwrite(img_path, resized)

    if needed <= 0:
        print(f"Target already met for {category}.")
        return

    print(f"Generating {needed} images into {inspect_path}...")
    generated_count = 0
    while generated_count < needed:
        img_name = random.choice(images)
        image = cv2.imread(os.path.join(path, img_name))
        if image is None: continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = augmenter(image=image)['image']
        
        # Save to inspection folder instead of BASE_DIR
        new_name = f"aug_{generated_count}_{img_name}"
        save_path = os.path.join(inspect_path, new_name)
        
        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
        generated_count += 1

for cat in CATEGORIES:
    augment_to_target(cat)

print(f"\nAll done! Check '{INSPECTION_DIR}' to review the new images before merging.")