"""
Blender script to import PLY point clouds or NPZ Gaussian data.
Usage in Blender:
    1. Open Blender
    2. Switch to Scripting workspace
    3. Open this script
    4. Modify the PLY_FILE or NPZ_FILE path
    5. Run script (Alt+P)

Or run from command line:
    blender --background --python blender_import_pointcloud.py
"""

import bpy
import numpy as np
import os
import sys
from mathutils import Vector, Quaternion


def import_ply_as_particles(ply_file, name="PointCloud"):
    """
    Import PLY point cloud as particle system in Blender.

    Args:
        ply_file: Path to PLY file
        name: Name for the object
    """
    print(f"Importing PLY: {ply_file}")

    # Import PLY
    bpy.ops.import_mesh.ply(filepath=ply_file)

    # Get the imported object
    obj = bpy.context.selected_objects[0]
    obj.name = name

    # Convert to point cloud visualization
    # Add particle system for better visualization
    ps = obj.modifiers.new("PointCloud", 'PARTICLE_SYSTEM')
    psys = ps.particle_system
    psys.settings.count = len(obj.data.vertices)
    psys.settings.emit_from = 'VERT'
    psys.settings.use_emit_random = False
    psys.settings.physics_type = 'NO'
    psys.settings.render_type = 'HALO'
    psys.settings.display_method = 'DOT'
    psys.settings.display_size = 0.003

    # Enable vertex colors if available
    if obj.data.vertex_colors:
        mat = bpy.data.materials.new(name=f"{name}_Material")
        mat.use_nodes = True
        obj.data.materials.append(mat)

        nodes = mat.node_tree.nodes
        nodes.clear()

        # Create nodes for vertex color display
        output = nodes.new('ShaderNodeOutputMaterial')
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        vcol = nodes.new('ShaderNodeVertexColor')

        vcol.layer_name = obj.data.vertex_colors[0].name

        # Connect nodes
        mat.node_tree.links.new(vcol.outputs[0], bsdf.inputs['Base Color'])
        mat.node_tree.links.new(bsdf.outputs[0], output.inputs[0])

    print(f"✓ Imported {len(obj.data.vertices)} points")
    return obj


def import_npz_as_instances(npz_file, name="Gaussians"):
    """
    Import NPZ Gaussian data as mesh instances.
    Each Gaussian is represented as a small sphere with position, rotation, and scale.

    Args:
        npz_file: Path to NPZ file
        name: Name for the collection
    """
    print(f"Importing NPZ: {npz_file}")

    # Load NPZ data
    data = np.load(npz_file)
    means = data['means']
    colors = data['colors']
    scales = data['scales']
    quaternions = data['quaternions']
    opacities = data['opacities']

    num_gaussians = len(means)
    print(f"Number of Gaussians: {num_gaussians}")

    # Create a collection for all Gaussians
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)

    # Create base sphere mesh (will be instanced)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.001, location=(0, 0, 0))
    base_sphere = bpy.context.active_object
    base_sphere.name = f"{name}_Base"

    # Sample every Nth Gaussian to avoid too many objects
    # Blender can handle ~10k objects, so sample accordingly
    sample_rate = max(1, num_gaussians // 5000)

    for i in range(0, num_gaussians, sample_rate):
        # Create instance of base sphere
        instance = base_sphere.copy()
        instance.data = base_sphere.data
        instance.name = f"{name}_{i}"

        # Set position
        instance.location = Vector(means[i])

        # Set rotation from quaternion (w, x, y, z)
        quat = Quaternion(quaternions[i])
        instance.rotation_mode = 'QUATERNION'
        instance.rotation_quaternion = quat

        # Set scale
        instance.scale = Vector(scales[i]) * 100  # Scale up for visibility

        # Create material with color
        mat = bpy.data.materials.new(name=f"{name}_Mat_{i}")
        mat.use_nodes = True
        mat.diffuse_color = (*colors[i], opacities[i])

        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Base Color'].default_value = (*colors[i], 1.0)
        bsdf.inputs['Alpha'].default_value = float(opacities[i])

        instance.data.materials.append(mat)

        # Add to collection
        collection.objects.link(instance)

    # Hide base sphere
    base_sphere.hide_set(True)
    base_sphere.hide_render = True

    print(f"✓ Created {num_gaussians // sample_rate} Gaussian instances")
    return collection


def import_animation_sequence(ply_dir, start_frame=0, frame_step=1):
    """
    Import sequence of PLY files as animation.

    Args:
        ply_dir: Directory containing PLY files (frame0000.ply, frame0001.ply, ...)
        start_frame: Starting frame number
        frame_step: Frame step (1 = every frame)
    """
    print(f"Importing animation sequence from: {ply_dir}")

    # Get all PLY files
    ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith('.ply')])

    if not ply_files:
        print("No PLY files found!")
        return

    print(f"Found {len(ply_files)} PLY files")

    # Create a collection for the animation
    collection = bpy.data.collections.new("Animation")
    bpy.context.scene.collection.children.link(collection)

    # Import each frame
    objects = []
    for i, ply_file in enumerate(ply_files[::frame_step]):
        ply_path = os.path.join(ply_dir, ply_file)

        # Import PLY
        bpy.ops.import_mesh.ply(filepath=ply_path)
        obj = bpy.context.selected_objects[0]
        obj.name = f"Frame_{i:04d}"

        # Add to collection
        collection.objects.link(obj)

        # Hide by default
        obj.hide_set(True)
        obj.hide_render = True

        # Keyframe visibility
        obj.hide_viewport = True
        obj.keyframe_insert(data_path="hide_viewport", frame=start_frame + i)
        obj.keyframe_insert(data_path="hide_render", frame=start_frame + i)

        if i > 0:
            obj.hide_viewport = False
            obj.keyframe_insert(data_path="hide_viewport", frame=start_frame + i - 1)
            obj.keyframe_insert(data_path="hide_render", frame=start_frame + i - 1)

        objects.append(obj)

    # Set animation range
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = start_frame + len(objects) - 1

    print(f"✓ Imported {len(objects)} frames")
    return objects


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Clear existing mesh objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # ========================================
    # Option 1: Import single PLY file
    # ========================================
    PLY_FILE = "/home/joon/dev/pose-splatter/output/markerless_mouse_nerf/pointclouds/frame0000.ply"

    if os.path.exists(PLY_FILE):
        import_ply_as_particles(PLY_FILE, name="MouseFrame0")

    # ========================================
    # Option 2: Import NPZ Gaussian data
    # ========================================
    # NPZ_FILE = "/home/joon/dev/pose-splatter/output/markerless_mouse_nerf/gaussians/gaussian_frame0000.npz"
    #
    # if os.path.exists(NPZ_FILE):
    #     import_npz_as_instances(NPZ_FILE, name="MouseGaussians")

    # ========================================
    # Option 3: Import animation sequence
    # ========================================
    # PLY_DIR = "/home/joon/dev/pose-splatter/output/markerless_mouse_nerf/animation/pointclouds"
    #
    # if os.path.exists(PLY_DIR):
    #     import_animation_sequence(PLY_DIR, start_frame=1, frame_step=1)

    print("\n✓ Import complete!")
    print("Tip: Adjust viewport shading to 'Solid' or 'Material Preview' to see colors")
