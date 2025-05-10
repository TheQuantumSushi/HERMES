# bulk_dataset.py

"""
Facilitates the generation of a dataset by automating the bulk generation of buildings
and bulk simulation of destructions.
"""

### IMPORT LIBRARIES :

import os
import subprocess
import time
import logging

### DEFINE PATHS :

HERMES_ROOT = os.environ["HERMES_ROOT"] # load from environment variable
BLENDER_EXE = os.environ["BLENDER_EXE"] # load from environment variable
LOG_PATH = os.path.join(HERMES_ROOT, "logs.txt")
SCRIPT_DIR = os.path.join(HERMES_ROOT, "scripts")
GEN_SCRIPT = os.path.join(SCRIPT_DIR, "generate_building.py")
SIM_SCRIPT = os.path.join(SCRIPT_DIR, "simulate_destruction.py")

### SETUP LOGGING :

logging.basicConfig(
    filename = LOG_PATH,
    level = logging.INFO,
    format = "%(asctime)s [bulk_dataset] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

### DEFINE TOOL FUNCTIONS :

def get_positive_int(prompt):
    """
    Keep asking for a prompt until a positive integer is given.

    Args :
        - prompt [str] : the prompt to ask the positive integer to the user

    Returns :
        - [int] : a positive integer
    """

    while True:
        try:
            value = int(input(prompt))
            if value < 0:
                print("Please enter a non-negative integer.")
            else:
                return value
        except ValueError:
            print("Invalid input type. Please enter an integer.")

### RUN THE GENERATION :

def main():
    """
    Ask for the number of generation and simulations to run and call the appropriate scripts.
    """

    # Initialize logging :
    start_time = time.time()
    logging.info("Script started")

    # Ask the user for the number of generations/simulations to run :
    n_generate = get_positive_int("Enter the number of buildings to generate : ")
    n_simulate = get_positive_int("Enter the number of simulations to run : ")
    logging.info(f"User input : n_generate = {n_generate}, n_simulate = {n_simulate}")

    for i in range(n_generate):
        t0 = time.time()
        logging.info(f"Generating building {i+1}/{n_generate}")
        subprocess.run([BLENDER_EXE, "--background", "--python", GEN_SCRIPT], check = True)
        logging.info(f"Generation {i+1} took {time.time()-t0:.2f}s")

    for i in range(n_simulate):
        t0 = time.time()
        logging.info(f"Simulating destruction {i+1}/{n_simulate}")
        subprocess.run([BLENDER_EXE, "--background", "--python", SIM_SCRIPT], check = True)
        logging.info(f"Simulation {i+1} took {time.time()-t0:.2f}s")

    # Log execution time of this program :
    duration = time.time() - start_time
    logging.info(f"All tasks completed successfully in {duration}s")

# Example usage :
if __name__ == "__main__":
    main()
