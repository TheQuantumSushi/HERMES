# cleanup.py

"""
Clean up the folders that may have been used (ask before, answer by yes/y, no/n) :
- HERMES/ai_hub/{all folders in here, but not files}
- HERMES/data/buildings
- HERMES/data/dataset/photos
- HERMES/data/dataset/labels.csv (this one is a file that gets cleared)
- HERMES/data/dataset/scenes
- HERMES/ml/models
- HERMES/ml/predictions
"""

### IMPORT LIBRARIES :

import os
import shutil
import time
import logging

### PATHS :

HERMES_ROOT = os.environ["HERMES_ROOT"] # Load root directory from environment
AIHUB_DIR = os.path.join(HERMES_ROOT, "ai_hub")
BUILDINGS_DIR = os.path.join(HERMES_ROOT, "data", "buildings")
PHOTOS_DIR = os.path.join(HERMES_ROOT, "data", "dataset", "photos")
LABELS_FILE = os.path.join(HERMES_ROOT, "data", "dataset", "labels.csv")
SCENES_DIR = os.path.join(HERMES_ROOT, "data", "dataset", "scenes")
MODELS_DIR = os.path.join(HERMES_ROOT, "ml", "models")
PREDICTIONS_DIR = os.path.join(HERMES_ROOT, "ml", "predictions")
LOG_PATH = os.path.join(HERMES_ROOT, "logs.txt")

### SETUP LOGGING :

logging.basicConfig(
    filename = LOG_PATH,
    level = logging.INFO,
    format = "%(asctime)s [cleanup] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

### DEFINE TOOL FUNCTIONS :

def prompt_yes_no(question):
    """
    Ask a question, which can be answered by yes/y or no/n.

    Args :
        - question [str] : the question to ask

    Returns :
        - bool : the boolean form of the answer, yes or no
    """

    while True:

        # Ask the question :
        ans = input(f"{question} (y/n) : ").strip().lower()

        # Get the answer :
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False

        # If answer is not valid, specify answer format :
        print("Please answer 'y' or 'n'")


def remove_path(path):
    """
    Remove the contents of a folder, or remove a specific file.

    Args :
        - path [str] : the path to the file/folder to delete
    """

    # Handle the file/folder not being found :
    if not os.path.exists(path):
        msg = f"{path} not found"
        print(msg)
        logging.warning(msg)
        return

    # If the file/folder is found, remove it :
    if os.path.isdir(path):
        shutil.rmtree(path)
        msg = f"Removed directory : {path}"
        print(msg)
        logging.info(msg)
    else:
        os.remove(path)
        msg = f"Removed file : {path}"
        print(msg)
        logging.info(msg)


def blank_file(path):
    """
    Delete the content of a file, not the file itself.

    Args :
        - path [str] : the path of the file to clear
    """

    # Handle the file not being found :
    if not os.path.isfile(path):
        msg = f"{path} not found"
        print(msg)
        logging.warning(msg)
        return

    # Truncate file to zero length :
    open(path, 'w').close()
    msg = f"Emptied file : {path}"
    print(msg)
    logging.info(msg)

### EXECUTE THE CLEANUP ACROSS ALL TARGETS :

def main():
    """
    Ask which targets to remove, then perform removals.
    """

    # Initialize logging :
    start_time = time.time()
    logging.info("Cleanup script started")
    print("Cleanup script will remove selected items.")

    # Prepare list of targets to clean after prompting :
    actions = []

    # Special handling of the ai_hub subfolders (appending them to the actions list) :
    if os.path.isdir(AIHUB_DIR):
        for entry in os.listdir(AIHUB_DIR):
            subpath = os.path.join(AIHUB_DIR, entry)
            if os.path.isdir(subpath):
                if prompt_yes_no(f"Clean up ai_hub subfolder '{entry}'?"):
                    actions.append(("remove", subpath))

    # Other, static targets :
    targets = [
        ("data/buildings folder", BUILDINGS_DIR),
        ("dataset/photos folder", PHOTOS_DIR),
        ("dataset/scenes folder", SCENES_DIR),
        ("ml/models folder", MODELS_DIR),
        ("ml/predictions folder", PREDICTIONS_DIR)
    ]
    for desc, path in targets:
        if prompt_yes_no(f"Clean up the {desc}?"):
            actions.append(("remove", path))

    # Special handling for labels.csv (which is a file, not a folder) :
    if prompt_yes_no("Clean up labels.csv ?"):
        actions.append(("blank", LABELS_FILE))

    # Perform all selected actions :
    for action, path in actions:
        if action == "remove":
            remove_path(path)
        elif action == "blank":
            blank_file(path)

    # Log execution time of this program :
    duration = time.time() - start_time
    logging.info(f"Cleanup script completed in {duration:.2f} seconds")

    print("Cleanup complete.")

# Example usage :
if __name__ == "__main__":
    main()
