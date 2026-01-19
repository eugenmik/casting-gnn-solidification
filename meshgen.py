import gmsh
import sys
import os
import glob
import math
import shutil
import time
from multiprocessing import Pool

# --- CONFIGURATION / SETTINGS ---
# Paths synchronized with project structure:
INPUT_DIR = "step"    # Folder containing procedural CAD files (.step)
OUTPUT_DIR = "data"    # Root folder for raw simulation data (.msh and .npz)
ERROR_DIR = "failed_meshes"

TARGET_ELEMENTS = 50000        
TOLERANCE_MM = 0.1             

# Adiabatic boundary settings
ADIABATIC_AXES = ["-Z"]       # Options: "+X", "-X", "+Y", "-Y", "+Z", "-Z"

# CPU Core settings for parallel processing
NUM_PROCESSES = 8  

# --------------------------------------------------------

def check_element_quality():
    """Returns (Passed, MinQuality). Returns False if element count is zero."""
    try:
        _, elem_tags, _ = gmsh.model.mesh.getElements(3)
        if len(elem_tags) == 0:
            return False, 0.0
        
        min_qual, max_qual = gmsh.model.mesh.getCustomQuality(3, 1) # SICN quality metric
        return min_qual > 0.01, min_qual
    except:
        return False, 0.0

def is_face_on_boundary(s_bb, model_bb, axis_code):
    """Detects if a surface face lies on the global bounding box boundary for adiabatic labeling."""
    gx_min, gy_min, gz_min, gx_max, gy_max, gz_max = model_bb
    sx_min, sy_min, sz_min, sx_max, sy_max, sz_max = s_bb
    
    tol = TOLERANCE_MM
    if axis_code == "+X": return abs(sx_max - gx_max) < tol and abs(sx_min - gx_max) < tol
    elif axis_code == "-X": return abs(sx_min - gx_min) < tol and abs(sx_max - gx_min) < tol
    elif axis_code == "+Y": return abs(sy_max - gy_max) < tol and abs(sy_min - gy_max) < tol
    elif axis_code == "-Y": return abs(sy_min - gy_min) < tol and abs(sy_max - gy_min) < tol
    elif axis_code == "+Z": return abs(sz_max - gz_max) < tol and abs(sz_min - gz_max) < tol
    elif axis_code == "-Z": return abs(sz_min - gz_min) < tol and abs(sz_max - gz_min) < tol
    return False

def get_mesh_size(target_count, volume_tag):
    """Calculates optimal mesh element size 'h' based on target element count and volume."""
    try:
        vol = gmsh.model.occ.getMass(3, volume_tag)
    except:
        vol = 0
    
    if vol <= 0.1: 
        return 0.0, 5.0
    
    h = (vol * 6.0 / target_count) ** (1.0/3.0)
    return vol, h

def process_file_wrapper(args):
    return process_single_file(*args)

def process_single_file(step_path, file_idx, total_files):
    filename = os.path.basename(step_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    suffix = ""
    if ADIABATIC_AXES:
        sorted_axes = sorted(ADIABATIC_AXES)
        suffix = "_" + "_".join(sorted_axes)
    
    output_filename = f"{name_no_ext}{suffix}.msh"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    gmsh.initialize()
    if NUM_PROCESSES > 1:
        gmsh.option.setNumber("General.Terminal", 0) 
    
    gmsh.model.add(name_no_ext)
    
    status = ""
    success = False

    try:
        gmsh.merge(step_path)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=3)
        if not volumes:
            raise Exception("No 3D Volume found")
        
        vol_tag = volumes[0][1]
        vol_val, h = get_mesh_size(TARGET_ELEMENTS, vol_tag)
        
        # Define physical groups for GNN markers (Adiabatic vs Cooling)
        model_bb = gmsh.model.getBoundingBox(-1, -1)
        adiabatic_tags = []
        cooling_tags = []
        surfaces = gmsh.model.getEntities(dim=2)

        for s in surfaces:
            tag = s[1]
            if not ADIABATIC_AXES:
                cooling_tags.append(tag)
            else:
                s_bb = gmsh.model.getBoundingBox(2, tag)
                is_adiabatic = False
                for axis in ADIABATIC_AXES:
                    if is_face_on_boundary(s_bb, model_bb, axis):
                        is_adiabatic = True
                        break
                if is_adiabatic: adiabatic_tags.append(tag)
                else: cooling_tags.append(tag)

        if adiabatic_tags: gmsh.model.addPhysicalGroup(2, adiabatic_tags, 1, "Adiabatic")
        if cooling_tags: gmsh.model.addPhysicalGroup(2, cooling_tags, 2, "Cooling")
        gmsh.model.addPhysicalGroup(3, [vol_tag], 3, "Volume")
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.7)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h * 1.3)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay
        
        gmsh.model.mesh.generate(3)
        
        _, elem_tags, _ = gmsh.model.mesh.getElements(3)
        elem_count = len(elem_tags[0]) if elem_tags else 0
        
        is_good, min_qual = check_element_quality()
        
        if success := (elem_count > 0):
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(output_path)
            status = f"OK (El:{elem_count}, Qual:{min_qual:.2f})"
            
    except Exception as e:
        status = f"FAIL: {str(e)}"
        success = False
    
    gmsh.finalize()
    return (filename, success, status)

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(ERROR_DIR): os.makedirs(ERROR_DIR)

    extensions = ["*.step", "*.STEP", "*.stp", "*.STP"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    files = sorted(list(set(files)))
    
    if not files:
        print(f"No files found in '{INPUT_DIR}' directory.")
        return

    print(f"--- Running Mesh Generation Pipeline (Cores: {NUM_PROCESSES}) ---")
    tasks = [(f, i, len(files)) for i, f in enumerate(files)]
    
    if NUM_PROCESSES == 1:
        results = [process_single_file(*t) for t in tasks]
    else:
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(process_file_wrapper, tasks)
    
    for filename, success, status in results:
        if not success:
            shutil.copy(os.path.join(INPUT_DIR, filename), os.path.join(ERROR_DIR, filename))

if __name__ == "__main__":
    main()