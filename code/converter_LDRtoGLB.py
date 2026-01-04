import os
import io
import trimesh
from trimesh.scene.lighting import PointLight, SpotLight
from trimesh.visual import TextureVisuals
import trimesh.visual
from trimesh.visual.material import SimpleMaterial
from trimesh.viewer import scene_to_html
import numpy as np
from PIL import Image
from pygltflib import GLTF2, PbrMetallicRoughness
import pyrender                                     #Need: PyRender needs: pip install numpy==1.26.4
#print(f"Trimesh version: {trimesh.__version__}")


"""
2025dec: DEZE CODE WERKT

+ Loads LDR-file
+ Incl. all meshes, textures & positions
+ Small file size

+ converter.export_glb() → Opens with Builder3D
+ converter.export_html_viewer → Opens with explorer, but dark, needs lights?

Optimalisations:
- Images should have size of 2^x.
- Some other issue's with 3d objects.
- Cylinder texture not right...

General: 
- Optimize generic light conditions?
TriMesh specific:
- scene.show(): does not show all faces somehow.
Export:
- Files size still big: optional re-scale images.
"""


# Set LDR units to meters for scale conversion
LDU_TO_METER = 0.0004


# ---------- LDraw parsing helpers ----------

def _load_ldconfig_colors(ldconfig_path):
    """Load colors from ldconfig.ldr and return a dict {color_code: [r,g,b,a]}."""
    colors = {}
    if not os.path.exists(ldconfig_path):
        return colors
    with open(ldconfig_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("0 !COLOUR"):
                parts = line.strip().split()
                try:
                    idx = int(parts[4])
                except:
                    continue
                rgb_hex = parts[6]
                r = int(rgb_hex[1:3], 16) / 255.0
                g = int(rgb_hex[3:5], 16) / 255.0
                b = int(rgb_hex[5:7], 16) / 255.0
                alpha = 1.0
                if "ALPHA" in parts:
                    a_idx = parts.index("ALPHA")
                    alpha = int(parts[a_idx+1]) / 255.0
                colors[idx] = [r, g, b, alpha]
    return colors

# ---------- Scene helpers ----------

def info(scene):
    print("=== SCENE ===")
    print(f"Scene bounds:      {scene.bounds}")
    print(f"Scene scale:       {scene.scale}")

    print("=== CAM ===")
    # Access the camera object
    if scene.camera is not None:
        camera = scene.camera
        # Get the camera's intrinsic matrix (K)
        # This includes focal length and resolution
        intrinsic_matrix = camera.K
        # Get the camera's transformation matrix (extrinsic parameters)
        # This is a (4, 4) homogeneous matrix that transforms from world space to camera space.
        camera_transform = scene.camera_transform
        camera_position = trimesh.transformations.translation_from_matrix(camera_transform)
        #camera_rotation = trimesh.transformations.rotation_from_matrix(camera_transform)

        # Get the camera's resolution in pixels (width, height)
        print(f"Camera Resolution: {camera.resolution}")
        # Get the camera's field of view (fovx, fovy) in degrees
        print(f"Camera FOV: {camera.fov}")
        print(f"Camera Position: {camera_position}")
        #print(f"Camera Transform:\n{camera_transform}")
    else:
        print("No camera found.")

    print("=== LIGHT ===")
    # Access the list of light objects
    lights = scene.lights
    # Use scene.graph.get to reliably get the transform for a named node
    for i, light in enumerate(lights):
        light_transform, _ = scene.graph.get(light.name)
        light_position = trimesh.transformations.translation_from_matrix(light_transform)
        
        print(f"  Light {i}:")
        print(f"  Name: {light.name}")
        print(f"  Type: {type(light).__name__}")
        print(f"  Color (RGBA): {light.color}")
        print(f"  Intensity: {light.intensity}")
        print(f"  Position: {light_position}")

    if not lights:
        print("No lights found in scene.")
        
        # # Check for specific light types (e.g., PointLight, SpotLight)
        # if isinstance(light, trimesh.scene.lighting.PointLight):
        #     print("  Type:        PointLight")
        # elif isinstance(light, trimesh.scene.lighting.SpotLight):
        #     print("  Type:        SpotLight")
    print("=== end ===")

#==============================================================================
#class LDRtoGLBConverter
#==============================================================================
class LDRtoGLBConverter:
    """
    A class to convert LDraw LDR files to GLB format with a dedicated parser.
    """

    def __init__(self, ldraw_path, texture_path, ldconfig_path):
        """
        Initializes the converter.

        Args:
            ldraw_path (str): Path to the LDraw parts library.
            texture_path (str): Path to the textures directory.
            ldconfig_path (str): Path to the ldconfig.ldr file.
        """
        self.ldraw_path = ldraw_path
        self.texture_path = texture_path
        self.ldconfig_path = ldconfig_path
        #self.color_map = _load_ldconfig(self.ldconfig_path)
        self.color_map = _load_ldconfig_colors(self.ldconfig_path)
        self.mesh_library = {}  # Cache for loaded meshes
        self.texture_library = {}  # Cache for loaded textures
        self.material_library = {} # Cache for loaded materials, along with textures.
        self.scene = trimesh.Scene()
        
# ---------- LDraw PART parsing ----------
    def _parse_part_dat(self, part_name, model_color):
        """
        Parses a .dat file to extract facets and texture information.
        Returns a trimesh mesh or None if the part cannot be parsed.
        """
        if part_name in self.mesh_library:
            return self.mesh_library[part_name].copy()

        # Try to find the part file in the standard locations
        part_path = self._find_part_path(part_name)
        
        if not part_path:
            print(f"Warning: Part file not found: {part_name}")
            return None

        vertices = []
        faces = []
        face_colors = []
        texture_info = None

        with open(part_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                line_type = parts[0]
                
                # Line Type 3: Triangle Facet
                # 3 503 52.173913 -0.000000 -2.608696 52.173913 -0.000000 2.608696 52.173913 -154.434783 2.608696
                if line_type == '3':
                    if parts[1].startswith("0x"): #convert 24 bit (hex) to [R,G,B,A]
                        rgb_code = int(parts[1], 16)
                        R = (rgb_code >> 16) & 0xFF
                        G = (rgb_code >> 8) & 0xFF
                        B = rgb_code & 0xFF
                        color_rgba = [ R, G, B, 255 ]
                    else: #assume colorcode from lDConfig.ldr:
                        color_code = int(parts[1])  
                        if color_code == 16:
                            color_rgba = self.color_map.get(model_color, [0, 0, 0, 255])
                        else:
                            color_rgba = self.color_map.get(color_code, [0, 0, 0, 255])

                    v1 = np.array(list(map(float, parts[2:5])))
                    v2 = np.array(list(map(float, parts[5:8])))
                    v3 = np.array(list(map(float, parts[8:11])))

                    current_vertex_count = len(vertices)
                    vertices.extend([v1, v2, v3])
                    faces.append([current_vertex_count, current_vertex_count + 1, current_vertex_count + 2])
                    face_colors.append(color_rgba)

                # !TEXMAP meta-statement
                # 0 !TEXMAP START PLANAR      -52.1 -154.4 0.0   52.1 -154.4 0.0    -52.1    0.0 0.0       m_029_large_clock.png
                # 0 !TEXMAP START CYLINDRICAL   0.0    0.0 0.0    0.0 -610.0 0.0    116.0 -610.0 0.0  360  ledifice_d22.png
                elif line_type == '0' and len(parts) > 1 and parts[1] == '!TEXMAP':
                    if parts[2] == 'START':
                        texture_info = {
                            'type': parts[3].lower(),
                            'filename': parts[-1], 
                            'params': list(map(float, parts[4:-1])),
                        }
                    elif parts[1] == 'END':
                        # The texture has ended, use the gathered info
                        pass

        if not vertices:
            return None

        # Build a trimesh object from the parsed data
        mesh = trimesh.Trimesh(
            vertices=np.array(vertices),
            faces=np.array(faces)
        )
        
        if texture_info:
            self._apply_texture_and_uv(mesh, texture_info)
            mesh.visual.face_colors = np.array([255, 255, 255, 255], dtype=np.uint8) * len(mesh.faces)
        else:
            # Fallback to color visuals if no texture is applied
            mesh.visual = trimesh.visual.ColorVisuals( face_colors=np.array(face_colors) )
        
        mesh.unmerge_vertices() #needed for getting rid of top-bright,bottom-dark effect.
        mesh.vertex_normals = None
        mesh.face_normals = mesh.face_normals

        # Cache the mesh and return a copy
        self.mesh_library[part_name] = mesh
        return mesh.copy()
    

    def _apply_texture_and_uv(self, mesh, texture_info):
        """
        Loads the texture and applies the correct UV mapping.
        """
        # Load and cache the texture
        texture_name = texture_info['filename']
        material=None
        if texture_name not in self.texture_library:
            texture_path = os.path.join(self.texture_path, texture_name)
            try:
                textureimage = Image.open(texture_path).convert('RGBA')
                if not texture_info['type'] == 'cylindrical':
                    textureimage = textureimage.transpose(Image.FLIP_TOP_BOTTOM)
                self.texture_library[texture_name] = textureimage 
                # IMPORTANT: trimesh assumes sRGB for baseColorTexture and to match LEOCAD more closely, add emissive:
                materialimage = trimesh.visual.material.PBRMaterial(
                    baseColorTexture=textureimage,
                    emissiveFactor=[0.2, 0.2, 0.2],
                    metallicFactor=0.0,
                    roughnessFactor=0.8 )
                self.material_library[texture_name] = materialimage

            except FileNotFoundError:
                print(f"Warning: Texture file not found: {texture_path}")
                self.texture_library[texture_name] = None
                self.material_library[texture_name] = None
                return
        
        texture = self.texture_library[texture_name]
        material = self.material_library[texture_name]
        if texture is None:
            return
        
        # Apply planar UV mapping
        if texture_info['type'] == 'planar':
            params = texture_info['params']
            p1 = np.array(params[0:3])
            p2 = np.array(params[3:6])
            p3 = np.array(params[6:9])

            # Simple planar projection based on the texture map's vectors
            v21 = p2 - p1
            v31 = p3 - p1
            
            vertices = mesh.vertices
            uvs = np.zeros((len(vertices), 2))

            for i, v in enumerate(vertices):
                vp = v - p1
                uvs[i, 0] = np.dot(vp, v21) / np.dot(v21, v21)
                uvs[i, 1] = np.dot(vp, v31) / np.dot(v31, v31)

            mesh.visual = TextureVisuals(uv=uvs, image=texture, material=material)

        # Apply cylindrical UV mapping
        elif texture_info['type'] == 'cylindrical':
            params = texture_info['params']

            p1 = np.array(params[0:3])   # axis start
            p2 = np.array(params[3:6])   # axis end
            p3 = np.array(params[6:9])   # reference point (defines seam)
            angle_deg = params[9]        # usually 360

            axis = p2 - p1
            axis_len = np.linalg.norm(axis)
            axis_dir = axis / axis_len

            # Build a stable orthonormal basis around the axis
            ref = p3 - p1
            ref -= axis_dir * np.dot(ref, axis_dir)
            ref /= np.linalg.norm(ref)

            ortho = np.cross(axis_dir, ref)

            vertices = mesh.vertices
            uvs = np.zeros((len(vertices), 2))

            for i, v in enumerate(vertices):
                vp = v - p1

                # ----- V: height along axis -----
                h = np.dot(vp, axis_dir)
                uvs[i, 1] = h / axis_len

                # ----- radial vector -----
                radial = vp - axis_dir * h
                r_len = np.linalg.norm(radial)
                if r_len < 1e-6:
                    uvs[i, 0] = 0.5
                    continue

                radial /= r_len

                # ----- U: angle around axis -----
                x = np.dot(radial, ref)
                y = np.dot(radial, ortho)

                theta = np.arctan2(y, x)   # [-π, π]
                u = theta / (2 * np.pi) + 0.5

                uvs[i, 0] = u

            mesh.visual = TextureVisuals(
                uv=uvs,
                image=texture,
                material=material
            )



# ---------- LDraw PART parsing ----------
    def _find_part_path(self, part_name):
        """
        Finds the full path to a part file, checking standard LDraw library locations.
        """
        for sub_dir in ['parts', 'p']:
            part_path = os.path.join(self.ldraw_path, sub_dir, part_name)
            if os.path.exists(part_path):
                return part_path
        return None
    
# ---------- LDraw LDR parsing ----------
    def _parse_ldr_file(self, file_path, parent_color):
        """
        Recursively parses an LDR file and its sub-parts.
        """
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            exit()
            #return trimesh.Scene()
        
        model_scene = trimesh.Scene()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                line_type = parts[0]                
                # Line Type 1: Sub-part Reference
                if line_type == '1':
                    #try:
                        color_code = int(parts[1])
                        #part_name = parts[14]
                        part_name = " ".join(str(x) for x in parts[14:])

                        current_color = color_code if color_code != 16 else parent_color
                        
                        # Extract and build transformation matrix
                        rotation_matrix = np.array(list(map(float, parts[5:14]))).reshape(3, 3)
                        translation_vector = np.array(list(map(float, parts[2:5])))
                        matrix = np.identity(4)
                        matrix[:3, :3] = rotation_matrix
                        matrix[:3, 3] = translation_vector

                        # Recursively parse the sub-part
                        sub_part_mesh = self._parse_part_dat(part_name, current_color)
                        if sub_part_mesh:
                            sub_part_mesh.apply_transform(matrix)
                            model_scene.add_geometry(sub_part_mesh)                             
                    #except (ValueError, IndexError) as e:
                    #    print(f"Warning: Skipping invalid line 1: {line.strip()} - {e}")
        
        return model_scene

# ---------- LDraw to scene conversion ----------
    def convert_ldr_to_scene(self, ldr_file_path):
        """
        Parses the main LDR file and builds the final trimesh scene.
        """
        self.scene = self._parse_ldr_file(ldr_file_path, 16) # Start with color 16 for the main model

        # Calculate global scale and rotation
        scale_matrix = trimesh.transformations.scale_matrix(LDU_TO_METER)
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        # Apply global transformation
        self.scene.apply_transform(np.dot(rotation_matrix, scale_matrix))

# ---------- Scene modifications ----------
    def clear_camera_and_lights(self, ClearCam=True,ClearLights=True):
        """
        Removes the camera and lights from the scene.
        """
        if ClearCam:
            self.scene.camera = None
        if ClearLights:
            self.scene.lights = []
            #del converter.scene.lights[1] #deletes item 1.

    def add_three_point_lighting(self, intensity=2.0, light_distance_multiplier=1.5):
        """
        Adds a three-point lighting setup to the scene, positioning the lights
        relative to the scene's bounding box.
        """
        if not self.scene.is_empty and self.scene.bounds is not None:
            bounds = self.scene.bounds
            center = self.scene.centroid
            max_dimension = np.linalg.norm(bounds - bounds)

            light_distance = max_dimension * light_distance_multiplier

            key_light = PointLight(intensity=intensity, name='key_light')
            fill_light = PointLight(intensity=intensity * 0.5, name='fill_light')
            back_light = PointLight(intensity=intensity * 0.7, name='back_light')

            key_transform = trimesh.transformations.translation_matrix(
                center + [light_distance, light_distance, light_distance]
            )
            fill_transform = trimesh.transformations.translation_matrix(
                center + [-light_distance, light_distance, -light_distance]
            )
            back_transform = trimesh.transformations.translation_matrix(
                center + [0, -light_distance, light_distance]
            )

            if hasattr(self.scene, 'add_light'):
                self.scene.add_light(key_light, key_transform)
                self.scene.add_light(fill_light, fill_transform)
                self.scene.add_light(back_light, back_transform)
            else:
                self.scene.lights.append(key_light)
                self.scene.lights.append(fill_light)
                self.scene.lights.append(back_light)

                # Determine the base frame for older versions
                base_frame_name = getattr(self.scene, 'base_frame', 'world')

                self.scene.graph.update(
                    frame_from=base_frame_name,
                    frame_to=key_light.name,
                    matrix=key_transform
                )
                self.scene.graph.update(
                    frame_from=base_frame_name,
                    frame_to=fill_light.name,
                    matrix=fill_transform
                )
                self.scene.graph.update(
                    frame_from=base_frame_name,
                    frame_to=back_light.name,
                    matrix=back_transform
                )
        else:
            print("Warning: Scene is empty, no lights were added.")


    def view_with_pyrender(self):
            """
            Converts the trimesh scene to a pyrender scene and displays it.
            Disables smooth shading to support face colors.
            """
            if self.scene.is_empty:
                print("Warning: Scene is empty and cannot be displayed.")
                return
            # Create a new, empty pyrender scene
            py_scene = pyrender.Scene()
            # Iterate through the trimesh scene's geometry and add to pyrender scene
            for name, geometry in self.scene.geometry.items():
                # Create a pyrender mesh, explicitly setting smooth=False
                py_mesh = pyrender.Mesh.from_trimesh(geometry, smooth=False)
                # Find the transform for this geometry from the trimesh scene's graph
                transform, _ = self.scene.graph.get(name)
                # Add the pyrender mesh to the pyrender scene with the correct transform
                py_scene.add(py_mesh, pose=transform)
            # Use a simple pyrender viewer with a default lighting setup
            pyrender.Viewer(py_scene, use_raymond_lighting=True)


# ---------- Viewer ----------
    def view_with_pyrender(self):
        """
        Converts the trimesh scene to a pyrender scene and displays it.
        Disables smooth shading and corrects orientation for pyrender.
        """
        if self.scene.is_empty:
            print("Warning: Scene is empty and cannot be displayed.")
            return

        py_scene = pyrender.Scene()

        # Define the orientation correction matrix (180-degree rotation around X-axis)
        # This converts from a Z-up like system to a Y-up system
        correction_matrix = trimesh.transformations.rotation_matrix(
            np.pi, [1, 0, 0]
        )

        for name, geometry in self.scene.geometry.items():
            py_mesh = pyrender.Mesh.from_trimesh(geometry, smooth=False)
            
            # Retrieve the transformation matrix from the trimesh scene
            transform, _ = self.scene.graph.get(name)

            # Apply the orientation correction to the transformation matrix
            corrected_pose = correction_matrix @ transform
            
            # Add the pyrender mesh with the corrected pose
            py_scene.add(py_mesh, pose=corrected_pose)

        # Add a camera for better viewing control
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        camera_pose = np.eye(4)
        # Position camera to look at the scene
        camera_pose[:3, 3] = self.scene.centroid + [0, 0, self.scene.scale * 2]
        py_scene.add(camera, pose=camera_pose)

        pyrender.Viewer(py_scene, use_raymond_lighting=True)


# ---------- Scene exports ----------
    def export_glb(self, output_file):
        """
        Exports the trimesh scene to a binary GLB file.
        """
        self.scene.export(output_file, file_type='glb', include_normals=True)
        print(f"GLB file exported successfully to {output_file}")

    def export_html_viewer(self, output_file="temp_viewer.html"):
        """
        Exports the scene to an HTML file for robust viewing in a web browser.
        """
        html_output = scene_to_html(self.scene)
        with open(output_file, "w") as f:
            f.write(html_output)
        print(f"Scene exported to {output_file} for viewing in browser.")



#==============================================================================
# MAIN:
#==============================================================================
if __name__ == '__main__':
    # Define paths    
    LDR_PARTS_LIBRARY = r'D:/Github/LDraw/_LDRAW_LIB'
    TEXTURES_DIR =  LDR_PARTS_LIBRARY + r'/parts/textures'
    LDCONFIG_FILE = r'D:/Github/LDraw/git/code/LDConfig.ldr'
    OUTPUT_GLB_PATH = r'D:/Github/LDraw/_GLB/'
    INPUT_GLB_PATH  = r'D:/Github/LDraw/_GLB/'


    #LDR_MODEL_NAME = r"ledifice_model_TR 23_Place_DHotel_De_Ville.ldr" 
    #LDR_MODEL_NAME = r"ledifice_model_HS_19 Palaisd'Aladin.ldr"  
    LDR_MODEL_NAME = r"ledifice_model_TR_09 Castel d'Elven.ldr"

    

    INPUT_LDR_FILE = INPUT_GLB_PATH + LDR_MODEL_NAME
    OUTPUT_GLB_FILE= OUTPUT_GLB_PATH + LDR_MODEL_NAME[:-3] + "glb"

    # Create the converter instance
    converter = LDRtoGLBConverter(LDR_PARTS_LIBRARY, TEXTURES_DIR, LDCONFIG_FILE)
    
    # Convert the LDR file to a trimesh scene:
    converter.convert_ldr_to_scene(INPUT_LDR_FILE)
    converter.clear_camera_and_lights()
    converter.add_three_point_lighting(intensity=2.0, light_distance_multiplier=2.5)
    #trimesh.util.concatenate(converter.scene) #only works for simular materials.

    try:                
        # View with TriMesh:
        #converter.scene.show(resolution=(1200, 800))
        # Issue: show(): → using mesh.append() works. → using model_scene.add_geometry() shows lot of missing faces, although textures do work.
        
        # View with PyRender:
        #converter.view_with_pyrender()                 #showing meshes fine, something wrong with textures...

        # Info scene:
        #info(converter.scene)

        # Export to HTML ready viewer:
        #converter.export_html_viewer(OUTPUT_GLB_PATH+"model_out.html")

        # Export the scene to a GLB file: WORKS:
        converter.export_glb(OUTPUT_GLB_FILE) #OUTPUT_GLB_PATH+"model_out.glb"

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

