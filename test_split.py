# A script to randomly select 50 images from each class (fire, smoke) to test folder
# This 100 images will serve as our test set.

import os
import random
import shutil

def create_test_set(source_dir, test_dir, category, num_samples=50):
    # Paths for the specific category
    src_path = os.path.join(source_dir, category)
    dest_path = os.path.join(test_dir, category)
    
    # Create the destination folder if it doesn't exist
    os.makedirs(dest_path, exist_ok=True)
    
    # Get all file names in the source folder
    all_files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
    
    # Error handling if you don't have enough images
    if len(all_files) < num_samples:
        print(f"Warning: Only {len(all_files)} images found in {category}. Selecting all.")
        num_samples = len(all_files)
    
    # Randomly select the files
    test_files = random.sample(all_files, num_samples)
    
    # Move the files
    for file_name in test_files:
        shutil.move(os.path.join(src_path, file_name), os.path.join(dest_path, file_name))
        
    print(f"Successfully moved {num_samples} images to {dest_path}")

# --- Configuration ---
SOURCE_DATA_DIR = 'C:/Users/charl/Downloads/Ghana-fire/train'  # Change this to your current folder\charl\Downloads\Ghana fire\train'  # Change this to your current folder path
TEST_DATA_DIR = 'C:/Users/charl/Downloads/Ghana-fire/test'    # Change this to where you want the test set
CATEGORIES = ['fire', 'smoke']

for cat in CATEGORIES:
    create_test_set(SOURCE_DATA_DIR, TEST_DATA_DIR, cat, num_samples=50)