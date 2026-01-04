# LDraw Model Library
Convert source date (STL/PNG) into LDraw/LeoCAD library files.
Load/Update/Save LDR/MPD files.
Convert LeoCAD models to GLB.
Convert source files to AI friendly input.

_________________________________________________
## Main library:
Basicall a part list, including file locations & metadata.

### Source files:
.csv; STL; picture files.

### class parts_db:
Collection of parts, consisting:
- Data:
  - STL file. (binairy) (1:1) [mm]
  - Texture file. (optional)
- Metadata: 
  - Part type.
  - Part Orientation: front, side, top
  - Orign of part: center, top, bottom, ..
  - RotationXaxis.
  - Pitch distance.
- LDraw export information:
  - LDraw description. (so categories could be used)
  - LDraw category.
  - LDraw scale.
  - LDraw color.
- FileVersion.

→ Import database (.csv)
→ Export database (.csv)
→ ExportLDrawFiles(destination_path)


### class MeshData:
Reads STL files and able to export to:
- LDraw, including optional planar textures.

→ Read STL file
→ Write STL file
→ Orientate STL file (metadata)
→ Scale STL file
→ ExportMeshToLDraw(metadata)

ToDo:
- Add cylindrical textures
- Copy org texture to destination_path
- Optional: change picture into 72 DPI

_________________________________________________
## LDraw Model Manager:
Loads/saves LeoCAD LDR/MPD files.

### cls ModelManager:
- Dict. of (sub)models:
  - Dict ID: model name
  - Per submodel: list of parts, like LDraw definition.
→ Import (LDR/MPD) (remove doubles, remove empty groups, ...)
→ Update (filenames, positions) (needs LibraryManager)
→ Export (LDR/MPD)
→ ExportSubModelsToLDR


_________________________________________________
## GLB Exporter:
Needs:
- ModelManager: load LDraw model.
- LibraryManager: load source parts.

### cls ExporterToGLB:
→ LibraryManager.Import()
→ ExportToGLB(file)
  - Create new scene
  - Add submodel?
  - Add mesh (3D / material)

_________________________________________________
## AI Exporter:
Needs:
- LibraryManager: load source parts.

### cls ExporterToAI:
→ GeneratePartList
→ Generate3Dmeshes
→ GenerateTextures