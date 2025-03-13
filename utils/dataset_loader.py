import os
from PIL import Image
import numpy as np


def load_dataset(data_dir):
    real_dir = os.path.join(data_dir, 'training_real')
    fake_dir = os.path.join(data_dir, 'training_fake')
    # Folder name should be 'real'


    # Check if the directories exist
    if not os.path.exists(real_dir):
        print(f"Error: The directory {real_dir} does not exist.")
        raise FileNotFoundError(f"Real images folder '{real_dir}' not found.")

    if not os.path.exists(fake_dir):
        print(f"Error: The directory {fake_dir} does not exist.")
        raise FileNotFoundError(f"Fake images folder '{fake_dir}' not found.")

    real_images = []
    fake_images = []

    # Load real images
    for file in os.listdir(real_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            image = Image.open(os.path.join(real_dir, file)).convert('RGB')
            image = image.resize((224, 224))  # Resize image to 224x224
            real_images.append(np.array(image))

    # Load fake images
    for file in os.listdir(fake_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            image = Image.open(os.path.join(fake_dir, file)).convert('RGB')
            image = image.resize((224, 224))  # Resize image to 224x224
            fake_images.append(np.array(image))

    # Convert to numpy arrays
    X = np.array(real_images + fake_images)
    y = np.array([0] * len(real_images) + [1] * len(fake_images))  # 0 for real, 1 for fake

    # Split dataset into train and test (80/20 split)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test
