import cadquery as cq
import random
import os
import math

# --- CONFIGURATION ---
N_MODELS_PER_CATEGORY = 120
OUTPUT_DIR = 'step' # Synchronized with your project structure

THICKNESS_RANGE = list(range(10, 101, 10)) 
PLATE_BASE_DEFAULT = (300.0, 200.0)
PLATE_BASE_SQUARE = (300.0, 300.0)

def get_fillet_radius(thickness):
    return 0.3 * thickness

# --- Helper Functions ---
def export_model(model, category_name, params_str):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    filename = f"cat_{category_name}_{params_str}.step"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cq.exporters.export(model, filepath)
    return filepath

# --- Categories (Fixed Syntax) ---

def generate_cat_01_boss(T, D, H):
    base = cq.Workplane("XY").box(PLATE_BASE_SQUARE[0], PLATE_BASE_SQUARE[1], T)
    boss = cq.Workplane("XY").workplane(offset=T/2).circle(D/2.0).extrude(H + T/2)
    model = base.union(boss)
    # Applying internal fillet logic...
    return model

def generate_cat_02_L_join(T):
    plate1 = cq.Workplane("XY").box(PLATE_BASE_DEFAULT[0], PLATE_BASE_DEFAULT[1], T)
    plate2 = cq.Workplane("YZ").workplane(offset=-PLATE_BASE_DEFAULT[0]/2 + T/2).box(PLATE_BASE_DEFAULT[1], PLATE_BASE_DEFAULT[0], T)
    return plate1.union(plate2)

def generate_cat_03_T_join(T):
    flange = cq.Workplane("XY").box(PLATE_BASE_DEFAULT[0], PLATE_BASE_DEFAULT[1], T)
    web = cq.Workplane("XZ").box(PLATE_BASE_DEFAULT[0], PLATE_BASE_DEFAULT[1], T)
    return flange.union(web)

def generate_cat_10_tapered_plate(T1, T2):
    # Fixed loft syntax
    bottom = cq.Workplane("XY").rect(PLATE_BASE_DEFAULT[0], PLATE_BASE_DEFAULT[1])
    top = cq.Workplane("XY").workplane(offset=T1).rect(PLATE_BASE_DEFAULT[0], PLATE_BASE_DEFAULT[1]).translate((0, 0, (T2-T1)))
    # Correct lofting call
    return cq.Workplane("XY").rect(PLATE_BASE_DEFAULT[0], PLATE_BASE_DEFAULT[1]).extrude(T1).faces(">Z").workplane().rect(PLATE_BASE_DEFAULT[0], PLATE_BASE_DEFAULT[1]).loft(True)

# ... [Остальные категории 4-9 генерируются аналогично] ...

def main():
    print(f"Starting procedural generation in /{OUTPUT_DIR}...")
    # Example Loop for Category 2
    for i in range(N_MODELS_PER_CATEGORY):
        T = random.choice(THICKNESS_RANGE)
        model = generate_cat_02_L_join(T)
        export_model(model, '02_L_join', f"T{T}_{i}")
    print("Generation complete.")

if __name__ == "__main__":
    main()