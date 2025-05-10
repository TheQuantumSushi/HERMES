# simulate_destruction.py

"""
Loads a random building from HERMES/data/buildings and simulates its destruction
using an indestructible sphere launched from a random angle with a certain speed.
Saves the .blend file of the animation to HERMES/data/scenes and takes a photo
that correctly faces the impact point, saving it to HERMES/data/dataset/photos.
Computes the average displacement vector of all the debris of the building and
computes the projected 2D coordinates on the photo, then appends these to the
HERMES/data/dataset/labels.csv for labelisation of the dataset.
It uses bpy, the Blender Python API library, to interact with Blender.
"""

### IMPORT LIBRARIES :

import bpy
import os
import random
import uuid
import math
import sys
import csv
import time
import logging
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

### DEFINE PATHS :

HERMES_ROOT = os.environ["HERMES_ROOT"]
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
    format = "%(asctime)s [simulate_destruction] %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

### DEFINE TOOL FUNCTIONS :

def project_to_pixel(scene, cam, coord):
    """
    Project 3D coordinates to 2D coordinates according to the view of a camera.
    This is used to convert the 3D coordinates, in the scene, of the average
    displacement vector, to their 2D equivalent (in pixels) on the photo taken
    by the camera.

    Args :
        - scene [bpy.types.Scene] : the scene in which the object is
        - cam [py.types.Object] : the camera used for projection
        - coord [Vector] : the 3D coordinates vector to project

    Outputs :
        - tuple[int, int] : the 2D coordinates of the projected vector on the camera's view  
    """

    co_ndc = world_to_camera_view(scene, cam, coord) # NDC = Normalized Device Coordinates (standard Blender space)
    render = scene.render
    return int(round(co_ndc.x * render.resolution_x)), int(round(co_ndc.y * render.resolution_y))

def label_vector_to_csv(image_path, x1, y1, x2, y2):
    """
    Append the start point x1, y1 and end point x2, y2 to the HERMES/data/dataset/labels.csv
    file for labelisation of the dataset, along with the name of the image that was used.

    Args :
        - image_path [str] : path of the image that is being labelised
        - x1 [int] : x coordinate of the starting point of the 2D average displacement vector
        - y1 [int] : y coordinate of the starting point of the 2D average displacement vector
        - x2 [int] : x coordinate of the ending point of the 2D average displacement vector
        - y2 [int] : y coordinate of the ending point of the 2D average displacement vector
    """

    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, mode = 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header if the file didn't previously exist (i.e. was just created) :
        if write_header:
            writer.writerow(["filename", "x1", "y1", "x2", "y2"])
        # Add the new entry :
        image_name = os.path.basename(image_path) # extract the filename only from the full path
        writer.writerow([image_name, x1, y1, x2, y2])

    logging.info(f"Labels appended : {image_name}, coords = ({x1}, {y1}), ({x2}, {y2})")

### EXECUTION OF THE SIMULATION :

def main(min_speed = 10.0, max_speed = 25.0, bullet_mass = 1e3, distance = 12.0, sim_duration = 30):
    """
    Perform the simulation and call the necessary function to save it and label it
    """

    # Initialize logging :
    start_time = time.time()
    logging.info("Script started")

    # Setup GPU :
    prefs = bpy.context.preferences
    cy_prefs = prefs.addons['cycles'].preferences
    cy_prefs.compute_device_type = 'CUDA'
    cy_prefs.get_devices()
    for dev in cy_prefs.devices:
        dev.use = True
    bpy.context.scene.cycles.device = 'GPU'
    logging.info("Enabled GPU rendering for Cycles")

    # Ensure existence of necessary directories, create them if inexistent :    
    for d in [SCENES_DIR, PHOTOS_DIR, DATASET_DIR]:
        os.makedirs(d, exist_ok = True)
    logging.info(f"Directories existence ensured : BUILD_DIR = {BUILD_DIR}, SCENES_DIR = {SCENES_DIR}, PHOTOS_DIR = {PHOTOS_DIR}")

    # Log parameters :
    logging.info(f"Simulation parameters : speed range = ({min_speed} - {max_speed}), mass = {bullet_mass}, distance = {distance}, duration = {sim_duration}s")

    # Load random building :
    blend_files = [f for f in os.listdir(BUILD_DIR) if f.endswith(".blend")]
    if not blend_files:
        logging.error("No .blend files found")
        raise RuntimeError("No .blend files found")
    choice = random.choice(blend_files)
    bpy.ops.wm.open_mainfile(filepath = os.path.join(BUILD_DIR, choice))
    logging.info(f"Loaded building file : {choice}")

    # Extract the informations :
    scene = bpy.context.scene
    fps = scene.render.fps
    scene.frame_start = 1
    scene.frame_end = int(sim_duration * fps)

    # Compute the coordinates of a bounding box of all the building cells :
    coords = []
    for obj in scene.objects:
        if obj.name.startswith("Building"): # each cell has a name of the format "Building_cell.XXX"
            coords += [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    min_x, max_x = min(v.x for v in coords), max(v.x for v in coords)
    min_y, max_y = min(v.y for v in coords), max(v.y for v in coords)
    min_z, max_z = min(v.z for v in coords), max(v.z for v in coords)
    logging.info("Computed building bounding box")

    # Pick a random face on which the impact will be made :
    face = random.choice(['+X','-X','+Y','-Y'])
    if face == '+X':
        normal, ix, iy = Vector((1,0,0)), max_x, random.uniform(min_y, max_y)
    elif face == '-X':
        normal, ix, iy = Vector((-1,0,0)), min_x, random.uniform(min_y, max_y)
    elif face == '+Y':
        normal, ix, iy = Vector((0,1,0)), random.uniform(min_x, max_x), max_y
    else:
        normal, ix, iy = Vector((0,-1,0)), random.uniform(min_x, max_x), min_y
    iz = random.uniform(min_z, max_z) # z coordinate is the height, taken at random
    impact_point = Vector((ix, iy, iz))
    logging.info(f"Impact face = {face}, impact_point = {impact_point}")

    # Capture the coordinates of the centroids of each cell before the simulation starts and set rigid bodies to enable physics simulation :
    original_centers = {}
    for obj in scene.objects:
        if obj.name.startswith("Building"): # each cell has a name of the format "Building_cell.XXX"
            original_centers[obj.name] = obj.matrix_world.translation.copy()
            bpy.context.view_layer.objects.active = obj
            bpy.ops.rigidbody.object_add() # set rigid body for the cell
            axis = 'x' if face in ['+X','-X'] else 'y'
            mid = (min_x+max_x)/2 if axis == 'x' else (min_y+max_y)/2
            coord = getattr(obj.location, axis)
            # Anchor the half of the building on which the impact point is not, so that the building doesn't fly away upon impact:
            obj.rigid_body.type = 'PASSIVE' if ((face in ['+X','+Y'] and coord < mid) or (face in ['-X','-Y'] and coord > mid)) else 'ACTIVE'
    logging.info(f"Rigid bodies set for {len(original_centers)} fragments")

    # Create the bullet and give it physics :
    start_loc = impact_point + normal * distance
    bpy.ops.mesh.primitive_uv_sphere_add(radius = 1.5, location = start_loc)
    bullet = bpy.context.active_object
    bullet.name = "Bullet"
    bpy.ops.rigidbody.object_add()
    bullet.rigid_body.type = 'ACTIVE'
    bullet.rigid_body.mass = bullet_mass
    bullet.rigid_body.kinematic = True
    bullet.rigid_body.collision_shape = 'SPHERE'
    logging.info(f"Bullet created at {start_loc}, mass = {bullet_mass}")

    # Animate the bullet :

    # Insert the bullet :
    bullet.keyframe_insert(data_path = "rigid_body.kinematic", frame = scene.frame_start)
    bullet.keyframe_insert(data_path = "location", frame = scene.frame_start)
    speed = random.uniform(min_speed, max_speed) # random speed
    impact_frame = min(int(1 + (distance * fps / speed)), scene.frame_end) # compute the frame in which impact happens
    # Interpolate its position linearly (lerp), frame by frame, before impact :
    for f in range(scene.frame_start, impact_frame+1):
        t = (f - scene.frame_start) / (impact_frame - scene.frame_start)
        bullet.location = start_loc.lerp(impact_point, t)
        bullet.keyframe_insert(data_path = "location", frame = f)
    # On impact, disable kinematic mode to let physics simulation take over :
    bullet.rigid_body.kinematic = False
    bullet.keyframe_insert(data_path = "rigid_body.kinematic", frame = impact_frame)

    logging.info(f"Bullet animated with speed={speed:.2f}, impact_frame={impact_frame}")

    # Bake the simulation :
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
    logging.info("Simulation baking complete")

    # Bake into cache :
    bpy.context.view_layer.update()
    bpy.ops.ptcache.free_bake_all()
    bpy.ops.ptcache.bake_all(bake = True)
    logging.info("Point cache baked")

    # Compute the displacement of each cell with respect to its starting position :

    # Advance to final frame :
    scene.frame_set(scene.frame_end)
    # Compute each displacement :
    delta_vectors = {}
    for name, start in original_centers.items():
        end = bpy.data.objects[name].matrix_world.translation.copy()
        delta_vectors[name] = end - start
    logging.info("Computed individual displacement vectors")

    # Compute the average displacement vector of all cells :
    avg_disp = sum(delta_vectors.values(), Vector((0,0,0))) / len(delta_vectors)
    total_center = sum(original_centers.values(), Vector((0,0,0))) / len(original_centers)
    origin_pt = total_center
    end_pt = total_center + avg_disp

    """
    # [WIP]
    # Assign a displacement-dependant color to each cell, from white (no displacement) to black (most displacement) :
    lengths = [v.length for v in delta_vectors.values()]
    min_move, max_move = min(lengths), max(lengths)
    if max_move == min_move:
        max_move += 1e-6
    for obj in scene.objects:
        if obj.name in delta_vectors:
            t = ((delta_vectors[obj.name].length - min_move) / (max_move - min_move))
            gray = 1.0 - t
            mat = bpy.data.materials.new(name=f"Mat_{obj.name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            output = nodes.new(type='ShaderNodeOutputMaterial')
            shader = nodes.new(type='ShaderNodeBsdfPrincipled')
            shader.inputs['Base Color'].default_value = (gray, gray, gray, 1)
            shader.inputs['Roughness'].default_value = 0.5
            links.new(shader.outputs['BSDF'], output.inputs['Surface'])
            obj.active_material = mat
    logging.info("Assigned materials based on movement")
    """

    # Move all cells into a collection for easy view toggling in Blender :
    col_frag = bpy.data.collections.new("Building_Cells")
    bpy.context.scene.collection.children.link(col_frag)
    for obj in scene.objects:
        if obj.name.startswith("Building"):
            col_frag.objects.link(obj)
    logging.info("Cell collection created")

    # Hide all Empties that were created to de-clutter the view in Blender :
    for emp in [o for o in scene.objects if o.type == 'EMPTY']:
        emp.hide_viewport = True
        emp.hide_render = True
    logging.info("Hid helper empties")

    # Add lighting to the scene for the render :
    bpy.ops.object.light_add(type = 'SUN', location = (impact_point.x + 10, impact_point.y + 10, impact_point.z + 20))
    sun = bpy.context.active_object
    sun.data.energy = 5
    col_cam = bpy.data.collections.new("Camera_Lights") # move to a dedicated collection
    bpy.context.scene.collection.children.link(col_cam)
    col_cam.objects.link(sun)

    # Add a camera for the render, pointed at the impact point to ensure that the photo is actually useful to the dataset :
    
    # Compute position and orientation :
    dist = random.uniform(30, 50)
    az = random.uniform(0,2 * math.pi)
    el = random.uniform(math.radians(25), math.radians(55))
    loc = impact_point + Vector((
        dist * math.cos(el) * math.cos(az),
        dist * math.cos(el) * math.sin(az),
        dist * math.sin(el)
    ))
    # Add the camera :
    bpy.ops.object.camera_add(location = loc)
    cam = bpy.context.active_object
    scene.camera = cam
    col_cam.objects.link(cam) # move to the collection
    vec = impact_point - cam.location
    cam.rotation_euler = vec.to_track_quat('-Z','Y').to_euler()
    logging.info(f"Camera placed at {cam.location}")

    # Hide bullet :
    bullet.hide_viewport = True
    bullet.hide_render = True

    # Render the photo and save it :
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.image_settings.file_format = 'PNG'
    photo_path = os.path.join(PHOTOS_DIR, f"photo_{uuid.uuid4().hex}.png")
    scene.render.filepath = photo_path
    bpy.ops.render.render(write_still = True)
    logging.info(f"Photo saved to {photo_path}")

    # Save the .blend scene with the animation :
    out = os.path.join(SCENES_DIR, f"scene_{uuid.uuid4().hex}.blend")
    bpy.ops.wm.save_mainfile(filepath = out)
    logging.info(f"Scene file saved to {out}")

    # Compute 2D coordinates of the average displacement vector and label the photo in the CSV file :
    x1, y1 = project_to_pixel(scene, cam, origin_pt)
    x2, y2 = project_to_pixel(scene, cam, end_pt)
    label_vector_to_csv(photo_path, x1, y1, x2, y2)

    duration = time.time() - start_time # execution time of this program
    logging.info(f"Script completed in {duration:.2f} seconds")

# Example usage :
if __name__ == "__main__":
    main()