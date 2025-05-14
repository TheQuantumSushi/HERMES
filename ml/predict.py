# predict.py

"""
Uses the model to make a prediction. The prediction is done on a new image, not one that
is part of the dataset. This image is generated under a new folder in HERMES/predictions,
called prediction_{date and time}. In it are the .blend files for the building and the
simulation, the base photo (photo.png), the label to evaluate the accuracy of the prediction
(label.csv). There is also a folder, HERMES/predictions/prediction{date_and_time}/results,
which contains the result photo on which is drawn the expected and predicted vectors, as well
as a text file after the prediction, metrics.txt, which recaps the performance of the prediction.
"""

### IMPORT LIBRARIES :

import os
import sys
import subprocess
import csv
import glob
import shutil
import json
import time
import datetime
import logging
import math
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

### DEFINE PATHS :

HERMES_ROOT = os.environ["HERMES_ROOT"] # load from environment variable
BLENDER_EXE = os.environ["BLENDER_EXE"] # load from environment variable
LOG_PATH = os.path.join(HERMES_ROOT, "logs.txt")
SCRIPT_DIR = os.path.join(HERMES_ROOT, "scripts")
GEN_SCRIPT = os.path.join(SCRIPT_DIR, "generate_building.py")
SIM_SCRIPT = os.path.join(SCRIPT_DIR, "simulate_destruction.py")
DATA_DIR = os.path.join(HERMES_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
BUILDINGS_DIR = os.path.join(DATA_DIR, "buildings")
SCENES_DIR = os.path.join(DATA_DIR, "scenes")
PHOTOS_DIR = os.path.join(DATASET_DIR, "photos")
CSV_PATH = os.path.join(DATASET_DIR, "labels.csv")
MODELS_DIR = os.path.join(HERMES_ROOT, "ml", "models")
CONFIG_PATH = os.path.join(MODELS_DIR, "train_config.json")
PREDICTIONS_DIR = os.path.join(HERMES_ROOT, "ml", "predictions")

### SETUP LOGGING :

logging.basicConfig(
    filename = LOG_PATH,
    level = logging.INFO,
    format = "%(asctime)s [predict] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

### DEFINE MODEL CLASS :

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride = 2, padding = 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride = 1, padding = 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride = 1, padding = 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.regressor(self.features(x))

### DEFINE TOOL FUNCTIONS :

def run_blender(script_path):
    """
    Run a script in blender using its executable

    Args :
        - script_path [str] : Blender executable path
    """

    # Initialize logging :
    t0 = time.time()
    logging.info(f"Running Blender script : {script_path}")

    # Run the subprocess in the background :
    subprocess.run([BLENDER_EXE, "--background", "--python", script_path], check = True)

    # Log completion time :
    logging.info(f"Completed {script_path} in {time.time()-t0:.2f}s")

def get_latest_file(directory, pattern="*.blend"):
    """
    Get the most recent file that matches a pattern inside a folder.
    Default pattern is to match .blend files.

    Args :
        - directory [str] : the path to the directory in which to search
        - pattern [str] : the pattern to match

    Returns :
        - str : the path to the latest file that matches the pattern inside the directory
    """

    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")

    return max(files, key = os.path.getmtime)

def remove_last_csv_line(csv_path):
    """
    This script uses the already-existing scripts to generate and simulate the destruction
    of buildings (HERMES/scripts/generate_building.py and HERMES/scripts/simulate_destruction.py),
    but those append the corresponding label to the HERMES/data/dataset/labels.csv file.
    Therefore, this function is here to remove the last line of this .csv.
    This may be changed in the future, adding a parameter in the two other programs to be able
    to choose whether or not to save the label when calling them.

    Args :
        - csv_path [str] : the path to the .csv file from which to remove the last line

    Returns :
        - str : the content of the last removed line, for future use elsewhere
    """

    # Open file and read it :
    with open(csv_path, "r", newline = "") as f:
        lines = f.readlines()

    # Extract last line and delete it :    
    last = lines[-1]
    with open(csv_path, "w", newline = "") as f:
        f.writelines(lines[:-1])

     return last.strip()

### EXECUTE THE GENERATION AND PREDICTION :

def main():
    """
    Generate a new building/simulation/photo/label apart from the dataset, use the model to make
    a prediction on it, and save performances.
    """

    # Initialize logging :
    start_all = time.time()
    logging.info("Script started")

    # Generate new building and simulation in the background :
    run_blender(GEN_SCRIPT)
    run_blender(SIM_SCRIPT)

    # Extract the generated .blend files as being the latest ones in their folders :
    building_src = get_latest_file(BUILD_DIR, "building_*.blend")
    scene_src = get_latest_file(AFTERMATH_DIR, "aftermath_*.blend")

    # Remove the last line of HERMES/data/dataset/labels.csv :
    csv_last = remove_last_csv_line(CSV_PATH)
    photo_filename, x1_s, y1_s, x2_s, y2_s = csv_last.split(",")

    # Store label values :
    labels_values = [int(x1_s), int(y1_s), int(x2_s), int(y2_s)]
    photo_src = os.path.join(PHOTOS_DIR, photo_filename)
    logging.info(f"Identified files : building = {building_src}, scene = {scene_src}, photo = {photo_src}")

    # Initialize a directory for the prediction (HERMES/ml/predictions/prediction_{date and time}) :
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    pred_dir = os.path.join(PREDICTIONS_DIR, f"prediction_{date_str}")
    os.makedirs(pred_dir, exist_ok = True)

    # Move the building and simulation .blend files, and rename them :
    shutil.move(building_src, os.path.join(pred_dir, "building.blend"))
    shutil.move(scene_src, os.path.join(pred_dir, "scene.blend"))
    shutil.move(photo_src, os.path.join(pred_dir, "photo.png"))

    # Write the new, isolated label.csv, under HERMS/ml/prediction_{date and time}/label.csv, using previously stored label values :
    with open(os.path.join(pred_dir, "label.csv"), "w", newline = "") as cf:
        writer = csv.writer(cf)
        writer.writerow(["x1", "y1", "x2", "y2"])
        writer.writerow(labels_values)

    logging.info("Moved source files and wrote label.csv")

    # Load the model and its configuration (HERMES/ml/models/cnn_regressor.pth and HERMES/ml/models/train_config.json)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "cnn_regressor.pth"), map_location = device))
    with open(CONFIG_PATH) as cf:
        config = json.load(cf) # the configuration file contains the parameters used for training, to log them in the metrics.txt file after the prediction
    logging.info(f"Loaded model and training config: {config}")

    # Make the prediction :
    tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    img = Image.open(os.path.join(pred_dir, "photo.png")).convert("RGB")
    input_tensor = tf(img).unsqueeze(0).to(device)
    t_inf0 = time.time()
    with torch.no_grad():
        pred = model(input_tensor).cpu().squeeze().tolist()
    inf_time = time.time() - t_inf0
    pred_int = [int(round(v)) for v in pred]
    logging.info(f"Prediction: {pred_int} (inference time {inf_time:.2f}s)")

    # Compute the metrics (label-specific errors, point-specific errors, MSE and MAE) :
    diffs = [p - t for p, t in zip(pred_int, labels_values)] # errors between expected and predicted x1, y1, x2, y2
    mse = sum(d*d for d in diffs) / len(diffs)
    mae = sum(abs(d) for d in diffs) / len(diffs)
    norm_start = math.hypot(diffs[0], diffs[1]) # error between expected and predicted starting point (x1, y1)
    norm_end = math.hypot(diffs[2], diffs[3]) # error between expected and predicted ending point (x2, y2)

    # Annotate the image to display results and save it :

    # Create the HERMES/ml/predictions/prediction_{date and time}/results folder :
    results_dir = os.path.join(pred_dir, "results")
    os.makedirs(results_dir, exist_ok = True)
    # Load the image :
    out_img = os.path.join(results_dir, "results.png")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    # Draw the expected vector (ground truth) in green :
    draw.line((labels_values[0], labels_values[1], labels_values[2], labels_values[3]), fill = "green", width = 3)
    draw.text((labels_values[0], labels_values[1]-10), "labels_values", fill = "green", font = font)
    # Draw the predicted vector in red :
    draw.line((pred_int[0], pred_int[1], pred_int[2], pred_int[3]), fill = "red", width = 3)
    draw.text((pred_int[0], pred_int[1]-10), "PR", fill = "red", font = font)
    # Save the image :
    img.save(out_img)
    logging.info(f"Saved annotated result to {out_img}")

    # Write and save the metrics.txt file under HERMES/predictions/prediction_{date and time}/results:
    metrics_path = os.path.join(pred_dir, results, "metrics.txt")
    with open(metrics_path, "w", encoding = "utf-8") as mf:
        mf.write(".\n")
        mf.write("└── Results of prediction\n")
        mf.write(f"    ├── Date : {date_str}\n")
        mf.write(f"    ├── Inference time : {inf_time:.4f}\n")
        mf.write("    ├── Parameters :\n")
        mf.write(f"    │   ├── epochs : {config['epochs']}\n")
        mf.write(f"    │   ├── batch size : {config['batch_size']}\n")
        mf.write(f"    │   └── learning rate : {config['learning_rate']}\n")
        mf.write("    ├── Ground truth :\n")
        mf.write(f"    │   ├── x1, y1 : {labels_values[0]}, {labels_values[1]}\n")
        mf.write(f"    │   └── x2, y2 : {labels_values[2]}, {labels_values[3]}\n")
        mf.write("    ├── Predicted values :\n")
        mf.write(f"    │   ├── x1, y1 : {pred_int[0]}, {pred_int[1]}\n")
        mf.write(f"    │   └── x2, y2 : {pred_int[2]}, {pred_int[3]}\n")
        mf.write("    ├── Coordinate errors :\n")
        mf.write("    │   ├── Coordinate-specific :\n")
        mf.write(f"    │   │   ├── x1 error : {diffs[0]}\n")
        mf.write(f"    │   │   ├── y1 error : {diffs[1]}\n")
        mf.write(f"    │   │   ├── x2 error : {diffs[2]}\n")
        mf.write(f"    │   │   └── y2 error : {diffs[3]}\n")
        mf.write("    │   └── Norm :\n")
        mf.write(f"    │       ├── start point (x1, y1) : {norm_start:.4f}\n")
        mf.write(f"    │       └── end point (x2, y2) : {norm_end:.4f}\n")
        mf.write("    └── Evaluation :\n")
        mf.write(f"        ├── MSE : {mse:.4f}\n")
        mf.write(f"        └── MAE : {mae:.4f}\n")
    logging.info(f"Wrote metrics to {metrics_path}")

    # Log execution time of this program :
    total_time = time.time() - start_all
    logging.info(f"Script completed in {total_time:.2f} seconds")

# Example usage :
if __name__ == "__main__":
    main()
