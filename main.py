import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


# ✅ Load Pretrained ResNet Model
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Updated weight loading
    model.fc = nn.Linear(512, 2)  # Change final layer for binary classification
    return model


# ✅ Preprocess Image Function
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        return None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).to(dtype=torch.float32)  # Convert to float32
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# ✅ Load Model Function
def load_model(model_path="saved_models/trained_model.pth"):
    model = get_model()
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found! Train the model first.")
        return None
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    return model


# ✅ Predict Image Function
def predict_image(image_path):
    model = load_model()  # Load trained model
    if model is None:
        return "Error: Model not found!"

    image = preprocess_image(image_path)
    if image is None:
        return "Error: Image not found!"

    # Get prediction
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence_real = probs[0][0].item()
        confidence_fake = probs[0][1].item()

    # ✅ Balanced Confidence Thresholds
    if confidence_fake >= 0.70:
        return "Fake", round(confidence_fake * 100, 2)
    elif confidence_real >= 0.40:
        return "Real", round(confidence_real * 100, 2)
    else:
        return "Uncertain", round(max(confidence_real, confidence_fake) * 100, 2)


# ✅ Dataset Class
class ImageDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = [os.path.join(real_dir, fname) for fname in os.listdir(real_dir) if
                            fname.endswith(('png', 'jpg', 'jpeg'))]
        self.fake_images = [os.path.join(fake_dir, fname) for fname in os.listdir(fake_dir) if
                            fname.endswith(('png', 'jpg', 'jpeg'))]
        self.images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# ✅ Load Dataset Function
def load_dataset(real_dir, fake_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise FileNotFoundError("Dataset folders not found! Ensure 'training_real' and 'training_fake' exist.")

    dataset = ImageDataset(real_dir, fake_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


# ✅ Train Model Function
def train_model():
    real_dir = "data/training_real"
    fake_dir = "data/training_fake"

    train_loader, test_loader = load_dataset(real_dir, fake_dir)
    model = get_model()

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 0.5]))  # ✅ More weight to Real images
    optimizer = optim.Adam(model.parameters(), lr=0.0003)  # ✅ Slower learning rate for better learning

    num_epochs = 30  # ✅ More epochs for better learning
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), "saved_models/trained_model.pth")
    print("✅ Model trained and saved.")


# ✅ Main Execution
if __name__ == "__main__":
    print("Starting training...")
    train_model()
