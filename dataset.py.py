import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_data(data_dir, img_size=(64, 64), max_samples=5000):
    """
    Loads cat and dog images, resizes them, flattens them into 1D vectors for SVM.
    """
    all_paths = (
        glob.glob(os.path.join(data_dir, "**/*.jpg"), recursive=True) +
        glob.glob(os.path.join(data_dir, "**/*.jpeg"), recursive=True)
    )
    
    images = []
    labels = []
    
    count = 0
    for path in tqdm(all_paths, desc="Loading Images"):
        filename = os.path.basename(path).lower()
        folders = [part.lower() for part in path.split(os.sep)]
        
        if filename.startswith("cat.") or "cat" in folders:
            label = 0
        elif filename.startswith("dog.") or "dog" in folders:
            label = 1
        else:
            continue
            
        try:
            with Image.open(path) as img:
                # Convert to Grayscale or RGB, then resize
                img = img.convert("L").resize(img_size)  # Grayscale for lower memory usage
                img_array = np.array(img).flatten()      # Flatten 2D image to 1D vector
                
                images.append(img_array)
                labels.append(label)
                count += 1
                
                if max_samples and count >= max_samples:
                    break
        except Exception as e:
            continue

    X = np.array(images, dtype=np.float32) / 255.0  # Normalize pixel values [0, 1]
    y = np.array(labels)
    
    print(f"\nLoaded {len(X)} images successfully. Image shape: {X[0].shape}")
    return X, y

def get_train_test_splits(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits data into Train, Validation, and Test sets (e.g., 80/10/10).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - relative_val_size, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test