import os, csv
import mesh_converter as mesh_converter

#import struct
#import math
#from math import cos, sin
#import numpy as np

"""
Type	:          Convention:	                Example:
 Variable       lowercase_with_underscores	 user_age
 Function       lowercase_with_underscores	 get_user_data()
 Constant       ALL_CAPS	                     MAX_RETRIES
 Class	        CamelCase                    CustomerAccount
 Method	        lowercase_with_underscores	calculate_total()
 Private	        _prefix	                     _internal_counter
 Special	        dunder	                     __init__
 Module/Package	lowercase_with_underscores	my_package, utils.py

============================================================================== 
PartID; PartType; Selected; Orientation; STLfile; TextureFile; 
AutoAlignment; Reposition [X,Y,Z]; RotateAroundX; PitchDistance; PartDescription; 
LDraw_File (obsolete); LDraw_File_OLD; LDraw_Category; LDraw_Keywords; LDraw_PitchDistance; LDraw_ColorCode; LDraw_ColorName; 
PovRay_Pigments; PovRay_Textures;
ExportVersion; 

============================================================================== 
"""


#==============================================================================
#class PartsDB: managing parts library
#==============================================================================
class PartsDB:
    def __init__(self):
        self.Parts = {}
        self.Path_Source =              r'D:/Github/LDraw/_SOURCE/'
        self.Path_ExportLDrawFiles =    r'D:/Github/LDraw/_LDRAW_LIB/parts/'
        root = os.getcwd()        
        self.PartsLibrary =             os.path.join(self.Path_Source, "lib_construction_systems.csv") #root

    def count_parts(self):
        return len(self.Parts)

    #-- import_csv_library --
    def import_csv_library(self, libFile=None, readAll=True):
        if libFile == None:
            libFile = self.PartsLibrary
        else:
            libFile = os.path.join(self.Path_Source, libFile) 

        #Read CSV database file:
        with open(libFile, mode='r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            line_count = 0 
            part_count = 0            
            for row in csv_reader:
                if line_count == 0:
                    PartsHeader = row
                    line_count += 1
                elif row[0].strip() == "":
                    pass    #pass empty row
                else:
                    newpart = dict(zip(PartsHeader, row))
                    if (newpart['Selected']=='yes' or readAll ):                         
                        replacementlist = newpart['Reposition [X,Y,Z]'][1:-1].split(',') 
                        newpart['Reposition [X,Y,Z]'] =         [ float(replacementlist[0]), float(replacementlist[1]), float(replacementlist[2]) ] 
                        newpart['RotateAroundX'] =              float( newpart['RotateAroundX'] )
                        newpart["LDraw_PitchDistance"] =        float(newpart["LDraw_PitchDistance"])
                        newpart["PitchDistance"] =              float(newpart["PitchDistance"])
                        if newpart['PartID'] in self.Parts: 
                            print ('Duplicate ID found: ' + row[0])
                        else:
                            self.Parts[row[0]] = newpart 
                            part_count += 1
                    line_count += 1
                if line_count>2000:
                    print("!! import_csv_library: reached max. line count.")
                    break
        csv_file.close()
        print(f'>Total parts ({len(self.Parts)}), imported: {part_count}/{line_count} rows.')

    
    #-- export_csv_library --
    def export_csv_library(self, libExportFile='lib_construction_systems_EXP.csv'):
        #Exporting loaded database CSV, might be used for updating CSV file.
        if len(self.Parts)==0:
            print("→export_csv_library: No parts available.")
            return
        part_count = 0
        with open(libExportFile, mode='w', newline='') as ldraw_file:
            csv_writer = csv.writer(ldraw_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow( self.Parts[ list(self.Parts.keys())[0] ].keys() )
            for key, value in self.Parts.items():
                csv_writer.writerow( value.values() )
                part_count += 1
        ldraw_file.close()
        print(f'>Exported library with {part_count} parts.')


    #-- ExportPartsToLDraw --
    def export_parts_to_ldraw_directory(self, exportAll=False, exportPath=None):
        if exportPath==None:
            exportPath = self.Path_ExportLDrawFiles        
        part_count = 0
        for key, value in self.Parts.items():
            if value['Selected'].lower()=='yes' or exportAll:
                part3d = mesh_converter.MeshData(self.Path_Source, exportPath)
                part3d.export_mesh_to_ldraw_file( value )
                part3d._ExportLDrawPicture()
                part_count += 1

    #-- Show Info --
    def show_info(self):        
        print( f"> Parts in library: {self.count_parts()}." )

#==============================================================================
#class PartsDB: managing parts library
#==============================================================================
if __name__ == "__main__":
    parts_db = PartsDB()
    # Import default csv library:
    #parts_db.import_csv_library() #default csv file
    parts_db.import_csv_library('lib_ConstructionSystemsLEdifice_All_EXP.csv')

    # Show library information:
    #parts_db.show_info()

    # Export all loaded parts to new csv:
    #parts_db.export_csv_library()

    # Convert all SELECTED parts to default LDraw directory:
    # Pre-pare folder with path: /library-name/ldraw/parts/textures
    # LeoCAD reference to path=mobaco_v1.5_EXP20251118/ldraw/
    # LeoCAD reference to zip=mobaco_v1.5_EXP20251118.zip. This zip should contain the folder 'ldraw/'.

    parts_db.export_parts_to_ldraw_directory( exportAll=False, exportPath=None ) #copy to standard all-parts library.
    #parts_db.export_parts_to_ldraw_directory( exportAll=False, exportPath='D:/Github/LDraw/cults3d/mobaco_v1.5_EXP20251118/ldraw/parts/' )