# train.py

"""
Train a CNN model on the photos under HERMES/data/dataset/photos, using the labelisation
at HERMES/data/dataset/labels.csv.
Saves the training parameters (epochs, batch size, learning rate) in HERMES/ml/models/train_config.json
for later use elsewhere.
"""

### IMPORT LIBRARIES :

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import BuildingDataset
from sklearn.metrics import mean_squared_error
import logging

### SETUP LOGGING :

logging.basicConfig(
    filename = LOG_PATH,
    level = logging.INFO,
    format = "%(asctime)s [train] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

### DEFINE MODEL CLASS :

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 5, stride = 2, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.regressor(self.features(x))

### DEFINE TOOL FUNCTIONS :

def evaluate_model(model, loader, device):
    """
    Evaluate the MSE of the model after its training.

    Args :
        - model [torch.nn.Module] : the PyTorch model to evaluate
        - loader [torch.utils.data.DataLoader] : the data loader used
        - device [torch.device] : the device (CPU or GPU) on which to run the evaluation
    """

    # Initialize logging :
    logging.info("Evaluation started")

    # Perform evaluation :
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # Compute MSE :
    mse = mean_squared_error(y_true, y_pred)

    logging.info(f"Evaluation MSE : {mse:.4f}")

def train_model(ml_dir, epochs = 300, batch_size = 16, lr = 1e-3):
    """
    Train the model with certain parameters.

    Args :
        - ml_dir [str] : the path to the folder in which the model and ML-related scripts are
        - epochs [int] : number of epochs ("passes") for which to train the model
        - batch_size [int] : number of images to process at once, in one epoch
        - lr [int] : learning rate for regression
    """

    # Initialize logging :
    t0 = time.time()
    logging.info(f"Training started : epochs = {epochs}, batch_size = {batch_size}, lr = {lr}")

    # Save training parameters in HERMES/ml/models/train_config.json for later use elsewhere :
    config = {"epochs" : epochs, "batch_size" : batch_size, "learning_rate" : lr}
    config_path = os.path.join(ml_dir, "models", "train_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok = True)
    with open(config_path, "w") as cf:
        json.dump(config, cf, indent = 2)
    logging.info(f"Saved training config to {config_path}")

    # Create the dataset :
    dataset = BuildingDataset(ml_dir)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    # Initialize training :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Perform training :
    for epoch in range(1, epochs + 1):
        e0 = time.time()
        model.train()
        running_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(dataset)
        logging.info(f"Epoch {epoch}/{epochs} - Loss : {epoch_loss:.4f} - took {time.time()-e0:.2f}s")

    # Once training is completed, save the trained model in HERMES/ml/models/cnn_regressor.pth :
    model_path = os.path.join(ml_dir, 'models', 'cnn_regressor.pth') # subject to change in the future to support multiple models
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

    # Evaluate the model :
    evaluate_model(model, loader, device)
    logging.info(f"Training completed in {time.time()-t0:.2f}s")

# Example usage :
if __name__ == '__main__':
    ml_directory = os.path.dirname(os.path.abspath(__file__))
    train_model(ml_directory)
