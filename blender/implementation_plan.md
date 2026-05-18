# Refactoring Microstructure Generator

The goal of this refactor is to significantly improve the realism, scientific accuracy, and performance of the headless Blender procedural generator (`microstructure_generator.py`). The blocky grid tessellation will be replaced with precise Voronoi polygons, carbide spawning will follow physical rules, materials will be deeply procedural to match real micrographs, and rendering will be optimized for speed using EEVEE.

## User Review Required

> [!IMPORTANT]
> **Dependency on `scipy`**
> The planned approach uses `scipy.spatial.Voronoi` to calculate exact mathematical grain boundaries. This requires `scipy` to be available in Blender's bundled Python environment. If it isn't pre-installed in your deployment environment, we will either need to install it (`pip install scipy` into Blender's Python) or fall back to an entirely Geometry Nodes-based Voronoi approach. Please confirm if `scipy` is acceptable.

> [!WARNING]
> **Blender Version for Shader Nodes**
> Blender 4.1+ merged the "Musgrave" texture into the "Noise" texture node. I plan to use the `ShaderNodeTexNoise` (and `ShaderNodeTexWave`) for the Martensite lath material to ensure compatibility with modern Blender versions, unless you are using an older version (3.x or 4.0) that specifically requires `ShaderNodeTexMusgrave`.

## Open Questions

1. Are you targeting a specific Blender version (e.g., 3.6 LTS, 4.0, or 4.1+)?
2. Is it acceptable to use a `bmesh` extrusion/inset technique to create actual physical depth (3D grooves) for the etching relief at the grain boundaries, which Eevee's lighting will capture beautifully?
3. Can we assume `scipy` is or can be installed in the environment where this script runs?

## Proposed Changes

### Core Logic & Tessellation

#### [MODIFY] `blender/microstructure_generator.py`
- **True Voronoi Tessellation**: Replace the `GRID_STEPS` proxy logic with `scipy.spatial.Voronoi`.
    - Generate seed points for grains.
    - To create a perfectly bounded 10x10 square slab, mirror the original seeds across the boundaries (top, bottom, left, right) before passing them to `Voronoi`.
    - Extract the finite regions corresponding to the original seeds to generate exact polygonal n-gons.
    - Construct a `bmesh`, creating one flat n-gon face per grain.
    - Apply an inset/bevel operation via `bmesh` or modifiers to physically recess the grain boundaries, simulating the etching relief.
    - Separate the faces into distinct objects (or mesh islands) based on their assigned phase to allow per-phase materials.

### Physics-Informed Precipitate Placement

#### [MODIFY] `blender/microstructure_generator.py`
- **Carbide Spawning**: Replace the random uniform distribution.
    - Identify the edges (line segments) of the Voronoi polygons.
    - Filter edges that belong to grains categorized as Ferrite or Martensite.
    - Distribute the calculated `N_CARBIDES` randomly along these specific edges (using vector interpolation).
    - Introduce a slight random offset so they sit *on* or *near* the boundary, mimicking real precipitation.

### Advanced Procedural Shading

#### [MODIFY] `blender/microstructure_generator.py`
- **Martensite Material**: Construct a node tree using `ShaderNodeTexWave` mixed with `ShaderNodeTexNoise` to create high-frequency, intersecting lath structures. Map these to base color and roughness.
- **Austenite/Ferrite Materials**: Utilize `ShaderNodeObjectInfo`'s `Random` output (or Geometry Node `Random Per Island`) plugged into a `ShaderNodeValToRGB` (ColorRamp). This will give each individual grain a slightly different shade, simulating how different crystallographic orientations reflect light differently under a microscope.
- **Etching Relief**: Use the physical recessed boundaries created during the tessellation step. Enhance with a subtle global normal/bump map across the grains to simulate imperfect polishing.

### Optimization & Engine Switching

#### [MODIFY] `blender/microstructure_generator.py`
- **Engine Switch**: Change `scene.render.engine` from `CYCLES` to `BLENDER_EEVEE` (or `BLENDER_EEVEE_NEXT` depending on the Blender version).
- **Settings**: Enable Ambient Occlusion, Screen Space Reflections, and high-quality shadows to make the etching relief pop.
- **Performance**: This should reduce render time from ~10s to <2s while maintaining stunning visual quality.
- **Constraints**: Ensure the CLI argument parsing, input JSON schema, and `--background` execution remain completely untouched.

## Verification Plan

### Automated Tests
- Execute the updated script in headless mode using a sample JSON payload:
  `blender --background --python microstructure_generator.py -- '{"martensite_pct": 72.0, "ferrite_pct": 20.0, "carbide_pct": 6.0, "austenite_pct": 2.0, "grain_size_um": 25.0, "output_path": "test_render.png", "seed": 42}'`
- Verify that `test_render.png` is generated successfully within the time constraint (under 5 seconds).

### Manual Verification
- Review the output image `test_render.png` to confirm:
  - Exact, sharp Voronoi polygonal boundaries (no jagged grids).
  - Carbides concentrated along the boundaries of Martensite/Ferrite.
  - Lath texturing inside Martensite.
  - Distinct grayscale variations per grain for Ferrite/Austenite.
  - Visible etching relief (depth/shadowing) at the grain boundaries.
