# generate_building.py

"""
Generates a pseudo-"building" by creating a cube and fragmenting it into cells.
It uses bpy, the Blender Python API library, to interact with Blender, as well as
the Cell Fracture add-on : https://extensions.blender.org/add-ons/cell-fracture/
Saves the building .blend file to HERMES/data/buildings.
"""

### IMPORT LIBRARIES :

import bpy
import os
import random
import time
import logging
import re

### DEFINE PATHS :

HERMES_ROOT = os.environ["HERMES_ROOT"] # load from environment variable
LOG_PATH = os.path.join(HERMES_ROOT, "logs.txt")
SCRIPT_DIR = os.path.join(HERMES_ROOT, "scripts")
DATA_DIR = os.path.join(HERMES_ROOT, "data")
BUILDINGS_DIR = os.path.join(DATA_DIR, "buildings")
SCENES_DIR = os.path.join(DATA_DIR, "scenes")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
PHOTOS_DIR = os.path.join(DATASET_DIR, "photos")
CSV_PATH = os.path.join(DATASET_DIR, "labels.csv")

### SETUP LOGGING :

logging.basicConfig(
    filename = LOG_PATH,
    level = logging.INFO,
    format = "%(asctime)s [generate_building] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

### EXECUTION OF THE GENERATION :

def main(min_dim = 4.0, max_dim = 10.0, seed_count = 100, cell_scale = (0.005, 0.005, 0.005), recursion_depth = 1, recursion_limit = 1000, recursion_clamp = 5000, recursion_chance = 0.9):
    """
    Generate the building and save it.

    Args :
        - min_dim [float] : minimum dimension for a side of the building
        - max_dim [float] : maximum dimension for a side of the building
        - seed_count [int] : number of initial points for Cell fracturation
        - cell_scale [tuple[float, float, float]] : the scale of each cell (x, y, z)
        - recursion_depth [int] : how many times to recursively fracture each cell
        - recursion_limit [int] : maximum number of total recursive operations
        - recursion_clamp [int] : maximum total resulting number of cells
        - recursion_chance [float] : probability (0 to 1) of each iteration to spawn a new recursion
    """

    # Initialize logging :
    start_time = time.time()
    logging.info("Script started")

    # Ensure existence of necessary directories, create them if inexistent :
    os.makedirs(BUILDINGS_DIR, exist_ok = True)
    logging.info(f"Parameters : min_dim = {min_dim}, max_dim = {max_dim}, seed_count = {seed_count}, cell_scale = {cell_scale}, recursion_depth = {recursion_depth},  recursion_limit = {recursion_limit}, recursion_clamp = {recursion_clamp}, BUILDINGS_DIR = {BUILDINGS_DIR}")

    # Create a new scene :
    bpy.ops.wm.read_factory_settings(use_empty = True)
    logging.info("Factory settings loaded")

    # Create a cube for the initial building :
    bpy.ops.mesh.primitive_cube_add()
    bld = bpy.context.active_object
    dims = [random.uniform(min_dim, max_dim) for _ in range(3)]
    bld.dimensions = dims
    bld.location.z = dims[2] / 2
    bld.name = "Building"
    logging.info(f"Created cube with dimensions = {dims}")

    # Enable Cell Fracture addon :
    bpy.ops.preferences.addon_enable(module='object_fracture_cell')

    # Add particle system for seeds :
    ps_mod = bld.modifiers.new("FractureSeeds", type = 'PARTICLE_SYSTEM')
    psys = bld.particle_systems[-1]
    pset = psys.settings
    pset.count = seed_count
    pset.frame_start = 1
    pset.frame_end = 1
    pset.emit_from = 'VOLUME'
    pset.distribution = 'RAND'
    pset.use_even_distribution = True
    bpy.context.view_layer.update()
    logging.info(f"Added particle system with {seed_count} seeds")

    # Apply Cell Fracture to the initial cube :
    bpy.ops.object.add_fracture_cell_objects(
        source = {'PARTICLE_OWN'},
        source_limit = seed_count,
        source_noise = 0.5,
        cell_scale = cell_scale,
        recursion = recursion_depth,
        recursion_source_limit = recursion_limit,
        recursion_clamp = recursion_clamp,
        recursion_chance = recursion_chance,
        recursion_chance_select = 'SIZE_MIN',
        use_smooth_faces = False,
        use_sharp_edges = True,
        use_sharp_edges_apply = True,
        use_data_match = True,
        use_island_split = True,
        margin = 0.0002,
        material_index = 0,
        use_remove_original = True
    )
    logging.info("Cell fracture applied")

    # Delete the initial cube, no longer needed :
    orig = bpy.context.scene.objects.get("Building")
    if orig:
        bpy.data.objects.remove(orig, do_unlink=True)
        logging.info("Original cube removed")

    # Assign rigid bodies to all the cells :
    cells = [o for o in bpy.context.scene.objects if o.name.startswith("Building")]
    for cell in cells:
        cell.select_set(True)
        bpy.context.view_layer.objects.active = cell
        bpy.ops.rigidbody.object_add()
        rb = cell.rigid_body
        rb.type = 'ACTIVE'
        rb.mass = random.uniform(1.0, 5.0)
        rb.collision_shape = 'CONVEX_HULL'
        rb.friction = 0.8
        rb.restitution = 0.05
        rb.use_deactivation = True
        rb.use_start_deactivated = True
        rb.deactivate_linear_velocity = 0.02
        rb.deactivate_angular_velocity = 0.02
    logging.info(f"Assigned rigid bodies to {len(cells)} cells")

    # Physical constraints, i.e. how cells interact with each other (they aggregate and stick together, as if they were one object that could fracture) :
    for o in bpy.context.scene.objects:
        o.select_set(False)
    for cell in cells:
        cell.select_set(True)
    bpy.context.view_layer.objects.active = cells[0]
    bpy.ops.rigidbody.connect(
        con_type = 'FIXED',
        pivot_type = 'CENTER',
        connection_pattern = 'CHAIN_DISTANCE'
    )
    for o in bpy.context.scene.objects:
        if o.type == 'EMPTY' and o.rigid_body_constraint:
            c = o.rigid_body_constraint
            c.use_breaking = True
            c.breaking_threshold = random.uniform(10.0, 50.0) # random threshold for the ammount of force needed to separate two cells
            c.disable_collisions = False
    logging.info("Constraints created and breaking enabled")

    # Add an indestructible ground (a plane) :
    bpy.ops.mesh.primitive_plane_add(size = 500.0)
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.scale = (50.0, 50.0, 1.0)
    ground.location.z = 0.0
    bpy.ops.rigidbody.object_add()
    grb = ground.rigid_body
    grb.type = 'PASSIVE'
    grb.friction = 1.0
    grb.restitution = 0.0
    grb.mass = 1e6
    logging.info("Ground created and physics assigned")

    # Save .blend file :

    _pattern = re.compile(r"building_(\d{4})\.blend$") # pattern to match files like "building_1234.blend"
    # List and filter existing files :
    existing = [f for f in os.listdir(BUILDINGS_DIR) if _pattern.match(f)]
    # Get the next building_XXXX.blend number :
    if existing:
        existing.sort() # sort by alphanumeric order
        last = existing[-1]
        last_num = int(_pattern.match(last).group(1))
        new_num = last_num + 1
    else:
        # If there are no files, then initialize number to 0 :
        new_num = 0
    # Build new filename with zero-padded 4-digit number :
    filename = f"building_{new_num:04d}.blend"
    filepath = os.path.join(BUILDINGS_DIR, filename)
    # Save and log :
    bpy.ops.wm.save_mainfile(filepath = filepath)
    logging.info(f"Saved building to {filepath}")

    # Log execution time of this program :
    duration = time.time() - start_time
    logging.info(f"Script completed in {duration:.2f} seconds")

# Example usage :
if __name__ == "__main__":
    main()
