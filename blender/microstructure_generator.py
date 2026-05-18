"""
ALLOY IQ — Blender Microstructure Visualizer
=============================================
Run headless: blender --background --python microstructure_generator.py -- <json_args>

Input JSON (via --args after --):
{
  "martensite_pct": 72.0,
  "ferrite_pct":    20.0,
  "carbide_pct":    6.0,
  "austenite_pct":  2.0,
  "grain_size_um":  25.0,
  "output_path":    "/tmp/microstructure.png",
  "seed":           42
}
"""

import bpy
import bmesh
import json
import sys
import math
import random
from mathutils import Vector, Color

# ─── Parse CLI arguments ────────────────────────────────────────────────────
argv = sys.argv
try:
    args_idx = argv.index("--") + 1
    args = json.loads(argv[args_idx])
except (ValueError, IndexError, json.JSONDecodeError):
    # Default demo values
    args = {
        "martensite_pct": 65.0,
        "ferrite_pct":    25.0,
        "carbide_pct":    8.0,
        "austenite_pct":  2.0,
        "grain_size_um":  20.0,
        "output_path":    "microstructure_render.png",
        "seed":           42,
    }

random.seed(args.get("seed", 42))

# ─── Phase fractions ─────────────────────────────────────────────────────────
MARTENSITE = args.get("martensite_pct", 70) / 100.0
FERRITE    = args.get("ferrite_pct", 22)    / 100.0
CARBIDE    = args.get("carbide_pct", 6)     / 100.0
AUSTENITE  = args.get("austenite_pct", 2)   / 100.0
GRAIN_UM   = args.get("grain_size_um", 25)
OUTPUT     = args.get("output_path", "microstructure_render.png")

# ─── Scene reset ─────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat)

# ─── Materials ────────────────────────────────────────────────────────────────
def make_principled_material(name: str, base_color, roughness=0.6, metallic=0.9, emission=None):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Base Color"].default_value   = (*base_color, 1.0)
    bsdf.inputs["Roughness"].default_value    = roughness
    bsdf.inputs["Metallic"].default_value     = metallic

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

# Phase materials
mat_martensite = make_principled_material("Martensite", (0.08, 0.08, 0.10), roughness=0.75, metallic=0.95)
mat_ferrite    = make_principled_material("Ferrite",    (0.25, 0.25, 0.28), roughness=0.55, metallic=0.90)
mat_austenite  = make_principled_material("Austenite",  (0.55, 0.45, 0.20), roughness=0.40, metallic=0.85)
mat_carbide    = make_principled_material("Carbide",    (0.03, 0.03, 0.03), roughness=0.20, metallic=0.50)

# Grain boundary material (dark lines)
mat_boundary   = make_principled_material("GrainBoundary", (0.01, 0.01, 0.01), roughness=0.95, metallic=0.0)

PHASE_MATS    = [mat_martensite, mat_ferrite, mat_austenite]
PHASE_WEIGHTS = [MARTENSITE, FERRITE, AUSTENITE]

# ─── Voronoi grain tessellation ───────────────────────────────────────────────
# Normalize grain size to scene units (1 unit = 50 µm)
SLAB_SIZE   = 10.0        # 10×10 slab
N_GRAINS    = max(20, int(SLAB_SIZE**2 / (GRAIN_UM / 50.0) ** 2))
N_GRAINS    = min(N_GRAINS, 150)        # cap for render speed

# Generate seed points
seeds_2d = [(random.uniform(0, SLAB_SIZE), random.uniform(0, SLAB_SIZE))
             for _ in range(N_GRAINS)]

def nearest_seed(x: float, y: float) -> int:
    best_i, best_d = 0, float("inf")
    for i, (sx, sy) in enumerate(seeds_2d):
        d = (x - sx) ** 2 + (y - sy) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i

# Assign phase to each grain seed
cumulative = []
running = 0.0
for w in PHASE_WEIGHTS:
    running += w
    cumulative.append(running)

def grain_phase_material(grain_idx: int):
    r = (grain_idx * 2654435761) % 1_000_000 / 1_000_000  # deterministic hash
    for i, threshold in enumerate(cumulative):
        if r < threshold:
            return PHASE_MATS[i]
    return PHASE_MATS[-1]

# Build grain mesh with Voronoi-like Delaunay coloring
GRID_STEPS = 80          # resolution of the slab
step       = SLAB_SIZE / GRID_STEPS
half       = SLAB_SIZE / 2.0

# Create one thin box per grain using a proxy approach:
# We cast a grid and group faces by their nearest seed → one mesh object per phase
from collections import defaultdict

phase_verts  = defaultdict(list)
phase_faces  = defaultdict(list)

vert_offset = defaultdict(int)

for row in range(GRID_STEPS):
    for col in range(GRID_STEPS):
        x0 = col * step - half
        y0 = row * step - half
        x1, y1 = x0 + step, y0 + step
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

        gidx  = nearest_seed(cx + half, cy + half)
        phase = grain_phase_material(gidx)
        p_key = phase.name

        base = vert_offset[p_key]
        phase_verts[p_key].extend([
            (x0, y0, 0.0), (x1, y0, 0.0),
            (x1, y1, 0.0), (x0, y1, 0.0),
        ])
        phase_faces[p_key].append((base, base+1, base+2, base+3))
        vert_offset[p_key] += 4

# Instantiate one mesh object per phase material
phase_mat_map = {m.name: m for m in PHASE_MATS}
slab_objects  = []

for p_key, verts in phase_verts.items():
    mesh = bpy.data.meshes.new(f"Grain_{p_key}")
    mesh.from_pydata(verts, [], phase_faces[p_key])
    mesh.update()

    obj = bpy.data.objects.new(f"Grain_{p_key}", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(phase_mat_map[p_key])
    slab_objects.append(obj)

# ─── Carbide precipitates (particle system emulation via small spheres) ───────
N_CARBIDES = int(CARBIDE * 400)   # scale with carbide fraction
carbide_positions = [
    (random.uniform(-half, half), random.uniform(-half, half), 0.0)
    for _ in range(N_CARBIDES)
]
carbide_radius = 0.04 + GRAIN_UM / 2500.0

if N_CARBIDES > 0:
    # Create template sphere
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=carbide_radius)
    template = bpy.context.active_object
    template.name = "Carbide_Template"
    template.data.materials.append(mat_carbide)

    for pos in carbide_positions:
        dup = template.copy()
        dup.data = template.data.copy()
        dup.location = Vector(pos)
        bpy.context.collection.objects.link(dup)

    # Remove template after instancing
    bpy.data.objects.remove(template)

# ─── Grain boundary wireframe overlay ─────────────────────────────────────────
# Draw thin boundary lines between adjacent cells with different nearest-seed
boundary_verts = []
boundary_edges = []
bv_idx = 0

for row in range(GRID_STEPS):
    for col in range(GRID_STEPS):
        cx = col * step - half + step / 2
        cy = row * step - half + step / 2
        gidx = nearest_seed(cx + half, cy + half)

        for dc, dr in [(1, 0), (0, 1)]:
            nc, nr = col + dc, row + dr
            if 0 <= nc < GRID_STEPS and 0 <= nr < GRID_STEPS:
                ncx = nc * step - half + step / 2
                ncy = nr * step - half + step / 2
                ngidx = nearest_seed(ncx + half, ncy + half)
                if gidx != ngidx:
                    x0 = col * step - half + dc * step
                    y0 = row * step - half + dr * step
                    boundary_verts.extend([(x0, y0, 0.005), (x0, y0, 0.006)])
                    boundary_edges.append((bv_idx, bv_idx + 1))
                    bv_idx += 2

if boundary_verts:
    bm_mesh = bpy.data.meshes.new("GrainBoundaries")
    bm_mesh.from_pydata(boundary_verts, boundary_edges, [])
    bm_mesh.update()
    bm_obj = bpy.data.objects.new("GrainBoundaries", bm_mesh)
    bm_obj.data.materials.append(mat_boundary)
    bpy.context.collection.objects.link(bm_obj)

# ─── Lighting ─────────────────────────────────────────────────────────────────
# Key light
bpy.ops.object.light_add(type="AREA", location=(0, 0, 12))
key = bpy.context.active_object
key.data.energy = 800
key.data.size   = 8

# Fill light
bpy.ops.object.light_add(type="POINT", location=(-5, -5, 6))
fill = bpy.context.active_object
fill.data.energy = 300

# ─── Camera ───────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(0, 0, 13))
cam = bpy.context.active_object
cam.rotation_euler = (0, 0, 0)
cam.data.type       = "ORTHO"
cam.data.ortho_scale = SLAB_SIZE * 1.05
bpy.context.scene.camera = cam

# ─── World background ─────────────────────────────────────────────────────────
world = bpy.data.worlds["World"]
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.02, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.3

# ─── Render settings ──────────────────────────────────────────────────────────
scene = bpy.context.scene
scene.render.engine         = "CYCLES"
scene.cycles.samples        = 64
scene.cycles.use_denoising  = True
scene.render.resolution_x   = 1920
scene.render.resolution_y   = 1080
scene.render.image_settings.file_format = "PNG"
scene.render.filepath       = OUTPUT

# ─── Render! ──────────────────────────────────────────────────────────────────
bpy.ops.render.render(write_still=True)
print(f"[ALLOY IQ] Microstructure render saved to: {OUTPUT}")
