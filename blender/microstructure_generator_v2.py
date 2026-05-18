"""
ALLOY IQ — Blender Microstructure Visualizer V2
=============================================
Run headless: blender --background --python microstructure_generator_v2.py -- <json_args>

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
import numpy as np
from mathutils import Vector, Color

# Ensure SciPy is available for Voronoi
try:
    from scipy.spatial import Voronoi
except ImportError:
    print("[ERROR] SciPy is not installed in Blender's Python environment.")
    print("Please install it using: 'path/to/blender/python/bin/python -m pip install scipy'")
    sys.exit(1)

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
        "output_path":    "microstructure_render_v2.png",
        "seed":           42,
    }

random.seed(args.get("seed", 42))
np.random.seed(args.get("seed", 42))

# ─── Phase fractions ─────────────────────────────────────────────────────────
MARTENSITE = args.get("martensite_pct", 70) / 100.0
FERRITE    = args.get("ferrite_pct", 22)    / 100.0
CARBIDE    = args.get("carbide_pct", 6)     / 100.0
AUSTENITE  = args.get("austenite_pct", 2)   / 100.0
GRAIN_UM   = args.get("grain_size_um", 25)
OUTPUT     = args.get("output_path", "microstructure_render_v2.png")

# ─── Scene reset ─────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat)

# ─── Materials ────────────────────────────────────────────────────────────────
def make_principled_material(name: str, base_color, roughness=0.6, metallic=0.9):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    
    # Handle Blender 4.0+ Principled BSDF changes gracefully
    if "Base Color" in bsdf.inputs:
        bsdf.inputs["Base Color"].default_value   = (*base_color, 1.0)
    bsdf.inputs["Roughness"].default_value    = roughness
    bsdf.inputs["Metallic"].default_value     = metallic

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nodes, links, bsdf

def setup_random_orientation(mat, nodes, links, bsdf, color1, color2):
    obj_info = nodes.new("ShaderNodeObjectInfo")
    obj_info.location = (-500, 0)
    
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (*color1, 1.0)
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (*color2, 1.0)
    color_ramp.location = (-200, 0)
    
    links.new(obj_info.outputs["Random"], color_ramp.inputs["Fac"])
    if "Base Color" in bsdf.inputs:
        links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

# 1. Martensite (Lath structure via procedural nodes)
mat_martensite, nodes_m, links_m, bsdf_m = make_principled_material("Martensite", (0.08, 0.08, 0.10), roughness=0.75, metallic=0.95)
tex_wave = nodes_m.new("ShaderNodeTexWave")
tex_wave.wave_type = 'BANDS'
tex_wave.wave_profile = 'SIN'
tex_wave.inputs["Scale"].default_value = 80.0
tex_wave.inputs["Distortion"].default_value = 6.0
tex_wave.inputs["Detail"].default_value = 15.0
tex_wave.location = (-600, 200)

tex_noise = nodes_m.new("ShaderNodeTexNoise")
tex_noise.inputs["Scale"].default_value = 150.0
tex_noise.inputs["Detail"].default_value = 15.0
tex_noise.location = (-600, -200)

math_mix = nodes_m.new("ShaderNodeMath")
math_mix.operation = 'MULTIPLY'
math_mix.location = (-300, 0)
links_m.new(tex_wave.outputs["Color"], math_mix.inputs[0])
links_m.new(tex_noise.outputs["Fac"], math_mix.inputs[1])

color_ramp_m = nodes_m.new("ShaderNodeValToRGB")
color_ramp_m.color_ramp.elements[0].position = 0.0
color_ramp_m.color_ramp.elements[0].color = (0.04, 0.04, 0.06, 1.0)
color_ramp_m.color_ramp.elements[1].position = 1.0
color_ramp_m.color_ramp.elements[1].color = (0.16, 0.16, 0.22, 1.0)
color_ramp_m.location = (-100, 200)

links_m.new(math_mix.outputs["Value"], color_ramp_m.inputs["Fac"])
if "Base Color" in bsdf_m.inputs:
    links_m.new(color_ramp_m.outputs["Color"], bsdf_m.inputs["Base Color"])
links_m.new(math_mix.outputs["Value"], bsdf_m.inputs["Roughness"])

# 2. Ferrite (Random grain orientation)
mat_ferrite, nodes_f, links_f, bsdf_f = make_principled_material("Ferrite", (0.25, 0.25, 0.28), roughness=0.55, metallic=0.90)
setup_random_orientation(mat_ferrite, nodes_f, links_f, bsdf_f, (0.2, 0.2, 0.25), (0.4, 0.4, 0.45))

# 3. Austenite (Random grain orientation)
mat_austenite, nodes_a, links_a, bsdf_a = make_principled_material("Austenite", (0.55, 0.45, 0.20), roughness=0.40, metallic=0.85)
setup_random_orientation(mat_austenite, nodes_a, links_a, bsdf_a, (0.45, 0.35, 0.15), (0.65, 0.55, 0.30))

# 4. Carbide
mat_carbide, _, _, _ = make_principled_material("Carbide", (0.02, 0.02, 0.02), roughness=0.20, metallic=0.60)

PHASE_MATS    = [mat_martensite, mat_ferrite, mat_austenite]
PHASE_WEIGHTS = [MARTENSITE, FERRITE, AUSTENITE]

# ─── True Voronoi grain tessellation ─────────────────────────────────────────
SLAB_SIZE   = 10.0
N_GRAINS    = max(20, int(SLAB_SIZE**2 / (GRAIN_UM / 50.0) ** 2))
N_GRAINS    = min(N_GRAINS, 300)

# 1. Base seeds
seeds = np.random.uniform(-SLAB_SIZE/2, SLAB_SIZE/2, size=(N_GRAINS, 2))

# 2. Mirror seeds across boundaries to perfectly clip the Voronoi to the slab
bounds = [-SLAB_SIZE/2, SLAB_SIZE/2]
mirrored_seeds = []
for x, y in seeds:
    mirrored_seeds.extend([
        [x, y],
        [bounds[0] - (x - bounds[0]), y],
        [bounds[1] + (bounds[1] - x), y],
        [x, bounds[0] - (y - bounds[0])],
        [x, bounds[1] + (bounds[1] - y)],
        [bounds[0] - (x - bounds[0]), bounds[0] - (y - bounds[0])],
        [bounds[1] + (bounds[1] - x), bounds[0] - (y - bounds[0])],
        [bounds[0] - (x - bounds[0]), bounds[1] + (bounds[1] - y)],
        [bounds[1] + (bounds[1] - x), bounds[1] + (bounds[1] - y)],
    ])

mirrored_seeds = np.array(mirrored_seeds)
vor = Voronoi(mirrored_seeds)

# Extract regions corresponding to original seeds
original_regions = [vor.point_region[i] for i in range(N_GRAINS)]

# Phase Assignment Logic
cumulative = []
running = 0.0
for w in PHASE_WEIGHTS:
    running += w
    cumulative.append(running)

def grain_phase_material():
    r = random.random()
    for i, threshold in enumerate(cumulative):
        if r < threshold:
            return PHASE_MATS[i], i
    return PHASE_MATS[-1], len(PHASE_MATS)-1

# 3. Build bmesh grains with etching relief
slab_objects = []
carbide_candidate_edges = []

for i, region_idx in enumerate(original_regions):
    region = vor.regions[region_idx]
    if -1 in region or len(region) == 0:
        continue
    
    polygon = vor.vertices[region]
    # Clamp slightly to ensure it stays within slab (handles float precision)
    polygon = np.clip(polygon, -SLAB_SIZE/2, SLAB_SIZE/2)
    
    mat, phase_idx = grain_phase_material()
    
    mesh = bpy.data.meshes.new(f"Grain_{i}")
    bm = bmesh.new()
    
    # Clean up degenerate vertices
    clean_poly = []
    for p in polygon:
        if not clean_poly or np.linalg.norm(p - clean_poly[-1]) > 1e-4:
            clean_poly.append(p)
    if len(clean_poly) > 1 and np.linalg.norm(clean_poly[0] - clean_poly[-1]) < 1e-4:
        clean_poly.pop()
        
    if len(clean_poly) < 3:
        bm.free()
        continue
        
    verts = [bm.verts.new((p[0], p[1], 0.0)) for p in clean_poly]
    try:
        face = bm.faces.new(verts)
        
        # Inset for etching relief: recess the boundary
        bmesh.ops.inset_region(bm, faces=[face], thickness=0.03, depth=0.0)
        for v in bm.verts:
            if v.is_boundary:
                v.co.z = -0.05
    except Exception:
        # Fallback if inset fails
        for v in bm.verts:
            v.co.z = -0.02
            
    bm.to_mesh(mesh)
    bm.free()
    
    # Auto-smooth to fix shading artifacts from n-gons
    mesh.use_auto_smooth = True
    mesh.auto_smooth_angle = math.radians(30)
    
    obj = bpy.data.objects.new(f"Grain_{i}", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    slab_objects.append(obj)
    
    # Collect edges for Physics-Informed Carbides (only Ferrite and Martensite)
    if phase_idx in [0, 1]:
        for j in range(len(clean_poly)):
            p1 = clean_poly[j]
            p2 = clean_poly[(j+1)%len(clean_poly)]
            carbide_candidate_edges.append((p1, p2))

# ─── Physics-Informed Carbides ───────────────────────────────────────────────
N_CARBIDES = int(CARBIDE * 400)
carbide_radius = 0.04 + GRAIN_UM / 2500.0

if N_CARBIDES > 0 and carbide_candidate_edges:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=carbide_radius)
    template = bpy.context.active_object
    template.name = "Carbide_Template"
    template.data.materials.append(mat_carbide)
    
    for _ in range(N_CARBIDES):
        edge = random.choice(carbide_candidate_edges)
        t = random.random()
        pos_x = edge[0][0] * (1-t) + edge[1][0] * t
        pos_y = edge[0][1] * (1-t) + edge[1][1] * t
        
        # Jitter so it clusters around boundary, settling in the etched groove
        pos_x += random.uniform(-0.03, 0.03)
        pos_y += random.uniform(-0.03, 0.03)
        pos_z = random.uniform(-0.04, 0.01) 
        
        dup = template.copy()
        dup.data = template.data.copy()
        dup.location = Vector((pos_x, pos_y, pos_z))
        
        # Random rotation for variety
        dup.rotation_euler = (random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14))
        # Random scale variation
        s = random.uniform(0.6, 1.4)
        dup.scale = (s, s, s)
        
        bpy.context.collection.objects.link(dup)

    bpy.data.objects.remove(template)

# ─── Lighting ─────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="AREA", location=(2, -2, 12))
key = bpy.context.active_object
key.data.energy = 1000
key.data.size   = 10

bpy.ops.object.light_add(type="POINT", location=(-5, 5, 8))
fill = bpy.context.active_object
fill.data.energy = 400

# ─── Camera ───────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(0, 0, 15))
cam = bpy.context.active_object
cam.rotation_euler = (0, 0, 0)
cam.data.type       = "ORTHO"
cam.data.ortho_scale = SLAB_SIZE * 1.05
bpy.context.scene.camera = cam

# ─── World background ─────────────────────────────────────────────────────────
world = bpy.data.worlds["World"]
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.01, 0.01, 0.01, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5

# ─── Render settings (Optimized for Eevee) ────────────────────────────────────
scene = bpy.context.scene
try:
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
except TypeError:
    scene.render.engine = 'BLENDER_EEVEE'

# Enable high-quality Eevee features if available
try:
    scene.eevee.use_gtao = True
    scene.eevee.use_ssr = True
    scene.eevee.taa_render_samples = 32
except AttributeError:
    pass

scene.render.resolution_x   = 1920
scene.render.resolution_y   = 1080
scene.render.image_settings.file_format = "PNG"
scene.render.filepath       = OUTPUT

# ─── Render! ──────────────────────────────────────────────────────────────────
bpy.ops.render.render(write_still=True)
print(f"[ALLOY IQ] Microstructure render v2 saved to: {OUTPUT}")
