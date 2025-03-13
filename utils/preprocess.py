from PIL import Image
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)  # Resize to the target size
    image = np.array(image) / 255.0    # Normalize the image
    return image
