# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 13:14:17 2025
@author: Edge

The mesh converter loads STL files.
Export to ldraw:
+ 3d: geometry: rotated, scaled (.DAT)
+ 2d: optional planar or cylindrical texture (.PNG)
"""

import json
import sys, os
from pathlib import Path
import numpy as np
import scipy
from PIL import Image
import math 
try:
    import trimesh
except Exception as e:
    print("trimesh is required. Install with: pip install trimesh", file=sys.stderr)
    raise
    

# ------------------------------
# Geometry & UV generation
# ------------------------------

def load_trimesh_from_stl(stl_file: Path) -> trimesh.Trimesh:
    #ONLY load binary meshes & doesnot read optional header.
    mesh = trimesh.load_mesh(str(stl_file), file_type='stl')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    if mesh is None:
        print("> !! load_trimesh_from_stl did not work.")
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    #mesh.remove_degenerate_faces()
    #mesh.remove_duplicate_faces()
    mesh.fix_normals()
    return mesh

def transform_trimesh(mesh: trimesh.Trimesh, 
                           rotate_x_deg: float = 0.0, 
                           align: str = None, 
                           translate_by_user: list = None,
                           scale: float = None):
    """
    Load STL with trimesh, center at origin, optionally rotate, auto-align to a face, optional user reposition & scale to LDraw.

    Args:
        mesh: refer to laoded mesh file
        rotate_x_deg (float): Degrees to rotate around the X-axis (default: 0).
        align (str): Auto-align option. One of:
                     'top', 'bottom', 'left', 'right', 'front', 'back', 'center'
        translate (list): Optional [x, y, z] translation after all transforms.
        scale (float): Optional scale the mesh.

        ToDo:
            texture_orientation (str): 'front', 'top', 'side' or 'round_up'.

    Returns:
        trimesh.Trimesh: The transformed mesh.
    """
    
    # --- 1. Center the STL at origin (bounding box center) ---
    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)

    # --- 2. Rotate around X-axis if requested ---
    if rotate_x_deg != 0:
        angle_rad = np.radians(rotate_x_deg)
        rot_matrix = trimesh.transformations.rotation_matrix(
            angle_rad, [1, 0, 0], point=[0, 0, 0]
        )
        mesh.apply_transform(rot_matrix)
        
    translate_vector = [0, 0, 0]
    # --- 3a Update new position when Auto-align to face ---
    if align is not None:
        bounds = mesh.bounds  # [[minx, miny, minz], [maxx, maxy, maxz]]
        min_corner, max_corner = bounds

        if align == "center":
            pass
        elif align == "top":     # move center top face to z=0
            translate_vector = [0, 0, -max_corner[2]]
        elif align == "bottom":  # move center bottom face to z=0
            translate_vector = [0, 0, -min_corner[2]]
        elif align == "left":   # move left face (x-min) to x=0
            translate_vector = [-min_corner[0], 0, 0]
        elif align == "right":  # move right face (x-max) to x=0
            translate_vector = [-max_corner[0], 0, 0]
        elif align == "front":  # move front face (y-max) to y=0
            translate_vector = [0, -max_corner[1], 0]
        elif align == "back":   # move back face (y-min) to y=0
            translate_vector = [0, -min_corner[1], 0]
        else:
            raise ValueError(f"Invalid align option: {align}")

    # --- 3b. Update new position by user if requested ---
    if translate_by_user is not None:
        for k in [0,1,2]:   #3:
            translate_vector[k] += translate_by_user[k]
    
    # --- 3c. Reposition if requested ---
    if translate_vector != [0, 0, 0]:
        mesh.apply_translation(translate_vector)

    # --- 4. Scale the mesh, according to LDraw scale factor ---
    if scale is not None and scale != 1.0:
        mesh = scale_trimesh(mesh, scale)

    return mesh


def scale_trimesh(mesh: trimesh.Trimesh, scale: float = None):
    meshx = mesh
    scale_matrix = np.eye(4) * scale
    scale_matrix[3, 3] = 1  # keep homogeneous coordinate
    mesh.apply_transform(scale_matrix)
    return meshx


def export_to_ldraw_dat(mesh: trimesh.Trimesh,
                        color: int = 16, 
                        min_edge_angle: float = 0,
                        texture_file: str = None,
                        texture_orientation: str = None,
                        scale: float = 0):
    """
    Export a trimesh mesh to an LDraw .dat library part file.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        color (int): LDraw colour code (default 16 = current).
        color (str): LDraw colour value 24bit RGB.
        min_edge_angle: None: no edges, else add unique edges, when angle>..
        texture_file (str): Optional texture file path.
        texture_orientation (str): 'front', 'top', 'side' & 'cylindrical_up'.
    """
    
    if scale!=0 or scale is not None:
        mesh = scale_trimesh(mesh, scale)
    bounds = mesh.bounds  # [[minx, miny, minz], [maxx, maxy, maxz]]
    min_corner, max_corner = bounds
    
    lines: list = []

    # build a reusable format specifier
    precision = ".3f"   # change to ".2f" or ".4f" easily
    fmt = f"{{:{precision}}}"
    
    # --- Optional TEXMAP ---
    if texture_file and texture_orientation:
        if texture_orientation == "front":  # STL → LDraw YZ plane: so swap Y↔-Z:
            p1, p2, p3 = [min_corner[0],-max_corner[2],0], [max_corner[0],-max_corner[2],0], [min_corner[0],-min_corner[2],0]
        elif texture_orientation == "top":  # STL → LDraw YZ plane: so swap Y↔-Z:
            #p1, p2, p3 = [0,0,0], [1,0,0], [0,0,1]
            p1, p2, p3 = [min_corner[0],0,min_corner[1]], [max_corner[0],0,min_corner[1]], [min_corner[0],0,max_corner[1]]
        elif texture_orientation == "side": # STL → LDraw YZ plane: so swap Y↔-Z:
            #p1, p2, p3 = [0,0,0], [0,1,0], [0,0,1]
            p1, p2, p3 = [0,-min_corner[2],min_corner[1]], [0,-max_corner[2],min_corner[1]], [0,-min_corner[2],max_corner[1]]
        elif texture_orientation == "cylindrical_up": # STL → LDraw YZ plane: so swap Y↔-Z:
            # 0 !TEXMAP START CYLINDRICAL 0 0 0   0 -305 0   49 -305 0   360 LEDIFICE_D22.png
            #print (bounds)
            p1, p2, p3 = [0,0,0], [0,-max_corner[2],0], [max_corner[0],-max_corner[2],0]
        else:
            raise ValueError(f"Invalid texture_orientation: {texture_file}, {texture_orientation}.")
        
        if texture_orientation in ['front','top','side']:
            lines.append(f"0 !TEXMAP START PLANAR "
                        f"{fmt.format(p1[0])} {fmt.format(p1[1])} {fmt.format(p1[2])} "
                        f"{fmt.format(p2[0])} {fmt.format(p2[1])} {fmt.format(p2[2])} "
                        f"{fmt.format(p3[0])} {fmt.format(p3[1])} {fmt.format(p3[2])} {texture_file[:-3].lower() +'png'}")   # Texture files are always converted to PNG.
                                                                                                                            #khw: added lowercase filename, not checked.
        elif texture_orientation in ['cylindrical_up']:
            lines.append(f"0 !TEXMAP START CYLINDRICAL "
                        f"{fmt.format(p1[0])} {fmt.format(p1[1])} {fmt.format(p1[2])} "
                        f"{fmt.format(p2[0])} {fmt.format(p2[1])} {fmt.format(p2[2])} "
                        f"{fmt.format(p3[0])} {fmt.format(p3[1])} {fmt.format(p3[2])} 360 {texture_file[:-3].lower()+'png'}")   # Texture files are always converted to PNG.
        lines.append("")

    # --- Faces (triangles) ---
    for face in mesh.faces:
        v1, v2, v3 = mesh.vertices[face]

        lines.append(
            f"3 {color} "
            f"{fmt.format(v1[0])} {fmt.format(-v1[2])} {fmt.format(v1[1])} "
            f"{fmt.format(v2[0])} {fmt.format(-v2[2])} {fmt.format(v2[1])} "
            f"{fmt.format(v3[0])} {fmt.format(-v3[2])} {fmt.format(v3[1])}" ) 
    lines.append("")
    
    # --- Feature edges ---    
    if min_edge_angle>0:

        # Returns: (n_pairs, 2) face indices and (n_pairs, 2) edge vertices
        adj_faces, adj_edges = trimesh.graph.face_adjacency(mesh.faces, return_edges=True)
        face_normals = mesh.face_normals

        min_angle_rad = np.deg2rad(min_edge_angle)
        
        for edge, faces in zip(adj_edges, adj_faces):            
            # Skip if not exactly two adjacent faces (i.e., boundary edge)
            if len(faces) != 2: # or np.any(faces == -1):
                continue
    
            n1, n2 = face_normals[faces[0]], face_normals[faces[1]]
            angle = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))

            if angle >= min_angle_rad:
                v1, v2 = mesh.vertices[edge]                
                lines.append(
                    f"2 24 "
                    f"{fmt.format(v1[0])} {fmt.format(-v1[2])} {fmt.format(v1[1])} "
                    f"{fmt.format(v2[0])} {fmt.format(-v2[2])} {fmt.format(v2[1])} " ) 
                    
    # --- End TEXMAP ---
    if texture_file and texture_orientation:
        lines.append("0 !TEXMAP END")
    
    return lines


def export_to_glb_part_definition(
    mesh: trimesh.Trimesh,
    texture_file: str | None = None,
    orientation: str = "front",
    rgba: tuple[int, int, int, int] = (200, 200, 200, 255)
) -> bytes:
    """
    Export a trimesh.Trimesh to GLB with either a texture or one constant RGBA color.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh geometry.
    texture_file : str, optional
        Path to texture image. If None, exports with a flat RGBA color.
    orientation : str
        Projection orientation for UV mapping ("front", "top", "side").
        Used only if texture_file is given.
    rgba : tuple of int
        (R, G, B, A) color in 0–255 for the whole mesh when no texture is provided.

    Returns
    -------
    bytes
        The GLB binary data.
    """
    mesh = mesh.copy()

    if texture_file:
        # Load texture image
        img = "" #Image.open(texture_file)

        # Generate UVs by projection
        v = mesh.vertices.copy()
        if orientation == "front":   # project onto X-Y plane
            uv = v[:, [0, 1]]
        elif orientation == "top":   # project onto X-Z plane
            uv = v[:, [0, 2]]
        elif orientation == "side":  # project onto Y-Z plane
            uv = v[:, [1, 2]]
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

        # Normalize UVs to [0,1]
        uv -= uv.min(axis=0)
        uv /= uv.ptp(axis=0)

        # Apply texture material
        material = trimesh.visual.texture.SimpleMaterial(image=img)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=img, material=material)

    else:
        # Flat color: assign per-face constant RGBA
        color_array = np.tile(np.array(rgba, dtype=np.uint8), (len(mesh.faces), 1))
        material = trimesh.visual.material.SimpleMaterial(
            name="flat_color",
            diffuse=tuple(rgba[:3])  # RGB only
        )
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=color_array, material=material)

    # Export as GLB
    return mesh.export(file_type="glb")


def texmap_size_px(x_mm):
    scale = 512 / 80  # 10.24 px per mm
    p = x_mm * scale
    #z = math.ceil(math.log2(p))  #Round up.
    z = math.floor(math.log2(p)) #Always round down
    #z = round(math.log2(p))      #Round to nearest power of two
    pixels = 2 ** z
    if pixels > 2048:
        pixels = 2048
    return pixels


#==============================================================================
#class MeshData: 
# 1. Reads original STL
# 2. Optional transforms the mesh (position, rotation), based on metadata
# 3. Exports (& optional scale) into LDraw format.
#==============================================================================
class MeshData():
    def __init__(self, source_path=None, destination_path=None):
        self.metadata = None
        self.source_path = source_path
        self.destination_path = destination_path
        self.mesh = None                            #Current mesh data.
        self.colorcode = None                       #16 → user color (seem model), else part has fixed 24bit RGB color
        self.LDrawScale = 2.5           #Scale in LDraw are in LDU's: 1 LDU = 0.4 mm
    
    def import_mesh(self, perform_reorientation=True):
        #Loads a bineary stl_file, with metadata
        #Expected coordinate system: right hand rule, Z+ is up.        
        sourcefile = os.path.join(self.source_path, self.metadata['STLfile'])
        #try:
        self.mesh = load_trimesh_from_stl( sourcefile )
        #except Exception as X:
        #    print(X)
        
        #Re-orientate the mesh (no scaling), when needed:
        if perform_reorientation:            
            self.mesh = transform_trimesh ( self.mesh, self.metadata['RotateAroundX'], self.metadata['AutoAlignment'].lower(), self.metadata['Reposition [X,Y,Z]'], None)
        
    def export_mesh(self, stl_out=None):
        #Exports the optional re-orientated (not scaled) mesh to STL file.
        try:
            self.mesh.export(stl_out)
            print( f"> Mesh stored: {stl_out}." )
        except Exception as X:
            print(X)
            

    def export_mesh_to_ldraw_file(self, metadata=None):
        #Exports the transformed mesh to LDraw file, including: metadata, scaling & optional texturing.
        if metadata is None:
            print ("!! Exporting part to LDraw. No metadata found.")
            return False

        self.metadata = metadata  # Store metadata in this class.
        self.import_mesh()

        if self.mesh == None:
            print ("!! Exporting part to LDraw: Could not load mesh.")
            return False

        lines = []

        # --- Header ---
        lines.append(f"0 {self.metadata['PartDescription']}")
        lines.append(f"0 Name: {self.metadata['PartID']}.dat")
        lines.append("0 Author: Koos Welling")        
        lines.append("0 !Unofficial Part")
        lines.append("0 !LICENSE Redistributable under CCAL version 2.0")
        lines.append("0 BFC CERTIFY CCW")  # assume counter-clockwise faces
        lines.append("")
        lines.append(f"0 !ORIGINALDATA: {self.metadata['STLfile']}." )
        #lines.append("0 !HISTORY updated {DATE}." )
        lines.append(f"0 !VERSION: {self.metadata['ExportVersion']}." )
        lines.append("0 // Created with parts_db.py & mesh_converter.py tools." )
        lines.append("0 // Parts used for virtual building with the vintage construction toys, like Mobaco." )
        lines.append("0 // This library part could be used in combination with LeoCAD." )
        lines.append(f"0 // System pitch [mm]: {self.metadata['PitchDistance']}" )
        lines.append(f"0 // LDraw  pitch [mm]: {self.metadata['LDraw_PitchDistance']}" )
        lines.append("0 // LDraw units: [ldu]." )
        lines.append("0 // See also: https://www.leocad.org/" )
        lines.append("0 // Check for latest vintage toy libraries: https://retrobuildingtoys.nl/" )
        lines.append("")
        if self.metadata['LDraw_ColorName'].startswith("0x"):
            lines.append(f"0 !COLOUR {self.metadata['LDraw_ColorName']} VALUE {self.metadata['LDraw_ColorCode']}" )
        else:
            lines.append(f"0 !COLOUR {self.metadata['LDraw_ColorName']} CODE {self.metadata['LDraw_ColorCode']}" )
        lines.append("")

        # --- add 3D content & optional texture file ---
        if len(self.metadata['TextureFile'])>0:
            texturefile = self.metadata['TextureFile']
        else: 
            texturefile = None
        
        scale_factor = self.LDrawScale * self.metadata["LDraw_PitchDistance"] / self.metadata["PitchDistance"] # mm→ldu * ldraw-snapping.
        lines += export_to_ldraw_dat(self.mesh, self.metadata['LDraw_ColorCode'], 25, texturefile, self.metadata.get('Orientation','front'), scale_factor )
        
        if texturefile != None:
            pass # There is no additional 3D data at end of texture.               

        # Save ldraw file (.dat)
        ldraw_out = os.path.join(self.destination_path, self.metadata['PartID'] + ".dat")
        with open(ldraw_out, "w", encoding="utf-8") as f:
            try:
                f.write("\n".join(lines))
                f.close()
                # if texturefile != None:
            except Exception as X:
                print(X)
                return False
            else:
                print(f"> Exported LDraw part: {self.metadata['PartID']}.dat")
                return True


    def _ExportLDrawPicture(self):
        if self.metadata is None:
            print ("> ! No metadata found, while converting picture.")
            return
        if len(self.metadata['TextureFile'])<5:
            return      # No texture used. 
        imagefile = str(self.source_path + self.metadata['STLfile'].split('/')[0] + "/_textures/" + self.metadata['TextureFile']) # expect picture inside: (stl-subfolder)+'_texuters'
        if not os.path.isfile(imagefile):
            print(f"> ! Could not find texture ({self.metadata['PartID']}): \n   ({imagefile})")
            return      # No texture found.
        if self.mesh is None:
            print(f"> ! No associated mesh found, while scaling {self.metadata['PartID']}.")
            return
        
        bounds = self.mesh.bounds  # [[minx, miny, minz], [maxx, maxy, maxz]] from MESH [ldu], in CAD coordinate system.
        size = np.subtract( bounds[1], bounds[0] )/2
        #print(f"> Mesh size:{size}.")
        if self.metadata["Orientation"]=="front": #x,z
            w, h = size[0], size[2]
        elif self.metadata["Orientation"]=="side": #y,z
            w, h = size[1], size[2]
        elif self.metadata["Orientation"]=="top": # x,y
            w, h = size[0], size[1]
        elif self.metadata["Orientation"]=="cylindrical_up":
            w, h = size[0]*3.14, size[1]
        else:
            print(f"> ! _ExportLDrawPicture: Error texture: {self.metadata['PartID']}. Wrong orientation: {self.metadata["Orientation"]}.")
        # Resize: standard panel: 50x80mm → 320x512pixels
        #wp = math.ceil(w*6.4/64)*64
        #hp = math.ceil(h*6.4/64)*64
        wp = texmap_size_px(w)
        hp = texmap_size_px(h)
        imageX = Image.open( imagefile )
        if imagefile.endswith('png'):
            # re-quantizeing color: 
            # if transparancy (alpha) is used: make sure this will be set to pallet index = 0.
            ConvertedImageX = imageX.quantize(colors=256,method=Image.FASTOCTREE)  
        else:
            ConvertedImageX = imageX.quantize(colors=256)  
        ConvertedImageX = ConvertedImageX.resize( (wp,hp), resample=0)   
        ConvertedImageX.save( self.destination_path + "textures/" + self.metadata['TextureFile'][:-3].lower() + "png", format='PNG') #khw: added lowercase filename, not checked.


    def export_render(self, view, border_px=25):
        #Exports on or more views as transparent PNG file, incl. border. Units are in mm & controlled by DPI value.
        pass
    
    def show_info(self):
        print (json.dumps(self.metadata, indent=4) )
        #print( self.metadata )
        



#==============================================================================
# Example use case.
#==============================================================================
if __name__ == "__main__":
    print("nothing to run.")