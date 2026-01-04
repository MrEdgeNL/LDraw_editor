# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:51:39 2022

@author: Edge

class LDrawSubModel():
    Stores actual (sub)Model data:
    + AddRowToSubmodel (incl: remove duplicates, optional Groups & Steps)
    + AddToBOM (BOM per submodel)
    + WriteToFileObj
    + ShowSubModelInfo
    + ShowModel    
    + _SubModelLimits [Xmin, Ymin, Zmin, Xmax, ..]
    + _SubModelSphere [X,Y,Z,r]
    

class LDrawModel():
    + GetKeyMainModel
    + _AddModelTxt

    + LoadLDrawModel     
    + LoadDirectory

    + SaveLDrawModel
    - ExportMPDtoLDR

    + ExportBOM (per submodel), necessary?
    + ShowModelInfo

    
LDraw Model Editor:
1. Loads MPD/LDR files
    Optional during file load: Remove duplicates from (every) LDR file. Dat doet ie toch bij submodel?
2. Update models: 
    + part positions (scale)
    + optional: update part names, needs instance of loaded parts_db

    
ToDo:
    - (Create random building(s)/cities)
    - (Create build plan (floorplan) mobaul-style)
    - (Create floor plan (based on collection))
"""

#import matplotlib.pyplot as plt
#import numpy as np
import os
from xmlrpc.client import boolean
from prettytable import PrettyTable
import parts_db


def RenameFiles(self, folder):
    #Helper function for re-naming files:
    for count, filename in enumerate(os.listdir(folder)):
        #dst = f"Hostel {str(count)}.jpg"
        #src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        #dst =f"{folder}/{dst}"
        if "1929" in filename:
            src = f"{folder}/{filename}" 
            dest = folder + filename.replace("1929", "S4")
            os.rename(src, dest) 



def UpdateBoundaryBox(cbb, newX, newY, newZ):
    if cbb==None:
        nBB = [newX, newY, newZ, newX, newY, newZ]
    else:
        nBB = cbb
        if newX<cbb[0]:
            nBB[0]=newX
        elif newX>cbb[3]:
            nBB[3]=newX
        elif newY<cbb[1]:
            nBB[1]=newY
        elif newY>cbb[4]:
            nBB[4]=newY
        elif newZ<cbb[2]:
            nBB[2]=newZ
        elif newZ>cbb[5]:
            nBB[5]=newZ        
    return nBB

# Applies correct matrix multiplication, for recursively expands all child submodels
def _mat_mul(A, B):
    """Multiply two 3x3 matrices"""
    return [
        [
            A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j]
            for j in range(3)
        ] for i in range(3)
    ]

def _mat_vec_mul(M, v):
    """Multiply 3x3 matrix with vector"""
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    ]

# --- helper function for exporting flattening:
def _submodel_starts_with_group(submodel):
    for row in submodel._Rows:
        if isinstance(row, str):
            if row.startswith("0 !LEOCAD GROUP BEGIN"):
                return True
            if row.strip() == "":
                continue
            if row.startswith("0"):
                continue
        break
    return False


#==============================================================================
#class LDrawSubModel: basically containing 1 LDR-model
#==============================================================================
class LDrawSubModel():
    def __init__(self, subModelName='New Model.ldr'):
        self._Name = subModelName
        self._Author = ''
        self._Rows = []
        self._NrOfDuplicates = 0
        self._NumberOfParts = 0
        self._NrOfSkippedSteps = 0
        self._NrOfSteps = 0                # Nr of remaining steps in submodel.
        self._NrOfSkippedGrps = 0
        self._BOM = {} 
        self._IsMainModel = True
        #self._PositionMatrix = 0            #OR matrix
        self._SubModelLimits = None            #None: not yet set, else: [xmin, ymin, .. , zmax]
        self._SubModelSphere = [0,0,0,0]    #Center & radius of sphere around submodel [X,Y,Z,Radius]
        self.LDrawScale = 2.5           #Scale in LDraw are in LDU's: 1 LDU = 0.4 mm
        
    
    
    def AddRowToSubmodel(self, txtRow):           
        #Just copy row, expecpt for: Duplicate parts; Empty STEPs; Empty GROUPs.
        if txtRow.startswith('0 Name:'):
            pass
        elif txtRow.startswith('0 Author:'):
            self._Author = txtRow.strip()[10:]  
        elif (txtRow.startswith('0 STEP')):
            if (len(self._Rows)>0) and (self._Rows[-1]==txtRow):        #if previous line was also a '0 STEP', skip this row:
                self._NrOfSkippedSteps += 1        
            else:
                self._Rows.extend( [ txtRow ] )
                self._NrOfSteps += 1
        elif txtRow.startswith('0 !LEOCAD GROUP END') and ( str(self._Rows[-1]).startswith('0 !LEOCAD GROUP BEGIN') ):        
                self._Rows.pop()
                self._NrOfSkippedGrps += 1 
        elif txtRow.startswith('1 '): #found new part (dat) or submodel (ldr):
            newpart = txtRow.split()[0:15]
            for i in range(0,2):
                newpart[i]=int(newpart[i])
            for i in range(2,14):
                newpart[i]=round(float(newpart[i]),2)
            newpart[14] = txtRow[txtRow.find(newpart[14]):].strip() 
            if newpart in self._Rows: #check for duplicates:
                self._NrOfDuplicates +=1
            else:
                self._Rows.extend( [ newpart ] )
                self._NumberOfParts += 1
                #self._AddToBOM(newpart[14])     #only create BOM when needed. 
        else:   #Nothing 'special, just add string to list', keeping file complete:
            self._Rows.extend( [ txtRow ] )
   
    
    def _AddToBOM(self, partname):
        if partname in self._BOM.keys():
            self._BOM[partname]=self._BOM[partname]+1
        else:
            self._BOM[partname]=1
    
    
    def CreateBOM(self):
        if len(self._BOM)==0:
            for row in self._Rows:
                if row[0]==1:
                    self._AddToBOM(row[14])
        return self._BOM
        
    
    def WriteToFileObj(self, fileout, addSubModelHeader:boolean, useSTEP:boolean, useGroups:boolean):
        if addSubModelHeader:
            fileout.write( '0 FILE ' + self._Name + "\n" )
        fileout.write( "0 " + "\n")
        fileout.write( "0 Name: " + self._Name + "\n")
        fileout.write( "0 Author: " + self._Author + "\n" )
        for rowX in self._Rows:
            if isinstance(rowX, list):      # Store DAT or LDR row:
                partline = ''.join(str(x)+' ' for x in rowX[0:15])
                fileout.write( partline + "\n" )
            else:                           # Add other info:
                if rowX.startswith('0 STEP') and (not useSTEP): #Optional skip STEP info
                    pass
                elif rowX.startswith('0 !LEOCAD GROUP') and (not useGroups): #Optional skip GROUPS info
                    pass
                else:
                    fileout.write( rowX +'\n' )
        if addSubModelHeader:
            fileout.write( '0 NOFILE\n' )        
    
    
    
    def ShowSubModelInfo(self):
        print ('   Number of objects:    {}'.format(self._NumberOfParts) )
        if self._NrOfDuplicates>0:   
            print ('   ! Skipped duplicates:   {}'.format(self._NrOfDuplicates) )  
        if self._NrOfSteps>0:   
            print ('   Steps used in model:  {}'.format(self._NrOfSteps+1) )
        if self._NrOfSkippedSteps>0:
            print ('   ! Nr. of skipped steps: {}'.format(self._NrOfSkippedSteps) )
        if self._NrOfSkippedGrps>0:
            print ('   ! Nr. of skipped GRPs:  {}'.format(self._NrOfSkippedGrps) )
        ##print ('   Local boundary box:')
        ##print ( self._SubModelLimits )
        #print ('   Local sphere:         ' + str(self._SubModelSphere) )
        
    def ShowModel(self):
        print(*self._Rows, sep = "\n")
        
    def ShowBOM(self):
        print( self._BOM )



#==============================================================================
#class LDrawModel: basically a MPD/LDR file, with potential submodels.
#==============================================================================
class LDrawModel():
    
    def __init__(self):
        self._SubModels = {} #DICT: { modelname: class_LDrawSubModel }        

    def clear(self):
        self._SubModels = {} #DICT: { modelname: class_LDrawSubModel }

    def GetKeyMainModel(self):
        return list(self._SubModels.keys())[0] #Most likely: "New Model.ldr"    

    def CountModels(self):
        return len(self._SubModels)

# ---------- Load/Save LDraw files ----------
    def LoadDirectory(self, Path):
        for x in os.listdir(Path):
            if x.endswith(tuple([".ldr", ".mpd"])):
                self.LoadLDrawModel( Path+x )
        
    def LoadLDrawModel(self, loadModelFileName):
        #Read MPD/LDR & parse text:
        f = open(loadModelFileName, 'r') # 'r' = read
        lines = f.readlines()
        f.close()
        #_filepath = os.path.dirname( loadModelFileName )
        _filename, fileextension = os.path.splitext( os.path.basename( loadModelFileName ) )
        #print( ._filepath + "\n" + _filename + "\n" + _fileextension)
        
        #Check if there are already models loaded
        if len(self._SubModels)>0:
            if not lines[0].startswith("0 FILE "):      #If single model (ldr), make it a submodel.
                firstSubModelName = "0 FILE " + os.path.basename( _filename ) #Don't use "New Model".
                lines.insert(0, firstSubModelName ) #pas op dat zelfde naamniet in dict komt, wat is de file naam?
                lines.insert(1, "0 ")
                lines.append("0 NOFILE")                
        self._AddModelTxt( lines )
    
    def _AddModelTxt(self, modellines, firstSubModelName=None):
        # Add model, both MPD/LDR & create submodels.
        # Remove duplicates, empty steps/groups.
        if firstSubModelName==None:
            smX = LDrawSubModel()                       #Incase of LDR, create default SubModel.        
        else:
            smX = LDrawSubModel(firstSubModelName )  
        for line in modellines:
            line = line.strip()
            if line.startswith("0 FILE "):          #Create new SubModel:
                if line[7:] in self._SubModels:
                    smX = LDrawSubModel( line[7:] + "_" + str(len(self._SubModels)) + "_" )
                else:
                    smX = LDrawSubModel( line[7:] )
            elif line.startswith("0 NOFILE"):       #Store new SubModel:
                self._SubModels[smX._Name] = smX 
            elif len(line)>2:                       #Add current line into current SubModel:
                smX.AddRowToSubmodel( line )
        if len(self._SubModels)==0:                 #Incase of LDR, store single SubModel
            self._SubModels[smX._Name] = smX
    
        
    def SaveLDrawModel(self, expModelFileName, useSTEP=True, useGroups=True):
        #Export MDR/LDR:
        with open(expModelFileName, mode='w', newline='') as fileout:
            for key in self._SubModels:             
                self._SubModels[key].WriteToFileObj( fileout, (len(self._SubModels)>1), useSTEP, useGroups )   
        print(f"> LDrawModel saved: {expModelFileName}.")
    
# ---------- Update LDraw files ----------
    def UpdateLDrawFilesFromDir(self, Path, preFix='', postFix='', scalefactor=1.0):
        # Update ALL ldraw files:
        for x in os.listdir(Path):
            x.lower()
            if x.endswith(".ldr") or x.endswith(".mpd"):
                self.UpdateLDrawfiles(Path,[x],preFix,postFix,scalefactor)
    
    def UpdateLDrawfiles(self, Directory, Files=[], preFix='', postFix='', scalefactor=1.0, partsDB=None):
        # Update LIST ldraw files:
        # → If partsDB: Rename parts in LDR files, based on changes in CSV database & scale
        # → Re-locate parts, depending on scale factor.
        PartNameToReplace = {}
        if partsDB is not None:
            for key, value in partsDB.Parts.items():
                if value['LDraw_File_OLD']:
                    PartNameToReplace[value['LDraw_File_OLD'].lower() ] = key + ".dat" #value['LDraw_File'].lower()
                if value['LDraw_File (obsolete)']:
                    PartNameToReplace[value['LDraw_File (obsolete)'].lower() ] = key + ".dat" #also rename the later "LDraw_File.dat"→ "{key}.dat"
            print(f'> Size of PartNamesToReplace: {len(PartNameToReplace)}.')
        for fileX in Files:
            with open(Directory+fileX, mode='r') as readFileX:
                Lines = readFileX.readlines()
                readFileX.close()
            filename, file_extension = os.path.splitext(fileX)
            changedparts_counter=0
            changedpositions_counter=0
            updateddata = ""
            for line in Lines:
                if line[0]=='1': #only lines refering to parts
                    # Read part pose line:
                    PartXpropertyList = line.split(" ")                    
                    PartXFilame = ( " ".join(PartXpropertyList[14:]) ).strip()
                    # REPLACE NAMES:
                    if (PartXFilame.lower() in PartNameToReplace):
                        PartXFilame = PartNameToReplace[PartXFilame.lower()] 
                        changedparts_counter += 1
                    
                    # UPDATE POSITIONS:
                    if (scalefactor != 1.0):
                        for k in [2,3,4]:
                            PartXpropertyList[k] = str( scalefactor * float(PartXpropertyList[k]) )
                        changedpositions_counter+=1
                    line = " ".join(PartXpropertyList[:14]) + " " + PartXFilame + "\n"
                updateddata += line #no parts? → just copy.
            if (changedparts_counter + changedpositions_counter)>0:
                with open(Directory+preFix+filename+postFix+file_extension, mode='w', newline='') as fileout:
                    fileout.write( updateddata )
                    fileout.close()
                print("> " + fileX + " → " + preFix+filename+postFix+file_extension + ":")
                print("  updated partnames = "+str(changedparts_counter))
                print("  updated positions = "+str(changedpositions_counter))

    def ExportMPDtoLDR(self, outdir, prefix, postfix, useSTEP=True, useGroups=True):
        # Strip MPD submodels into LDR files.        
        for key in self._SubModels:
            expModelFileName = os.path.join(outdir, prefix+key+postfix)
            expModelFileName = expModelFileName.replace(".ldr","",-1) + ".ldr"
            with open(expModelFileName, mode='w', newline='') as fileout:
                self._SubModels[key].WriteToFileObj( fileout, False, useSTEP, useGroups )   
                print(f"> LDrawModel saved: {expModelFileName}.")
            fileout.close()


    # Recursively expands all child submodels
    def _flatten_submodel(self, submodel_name, parent_pos, parent_rot, out_rows, is_root=False):
        sm = self._SubModels[submodel_name]

        # ---- GROUP BEGIN (if needed & not root) ----
        wrap_in_group = (
            not is_root and
            not _submodel_starts_with_group(sm)
        )

        if wrap_in_group:
            out_rows.append(f"0 !LEOCAD GROUP BEGIN grp {submodel_name}")

        for row in sm._Rows:
            if not isinstance(row, list):
                out_rows.append(row)
                continue

            color = row[1]
            pos = row[2:5]
            rot = [
                row[5:8],
                row[8:11],
                row[11:14]
            ]
            ref = row[14]

            # combine transforms
            new_pos = _mat_vec_mul(parent_rot, pos)
            new_pos = [
                new_pos[0] + parent_pos[0],
                new_pos[1] + parent_pos[1],
                new_pos[2] + parent_pos[2],
            ]
            new_rot = _mat_mul(parent_rot, rot)

            if ref in self._SubModels:
                self._flatten_submodel(ref, new_pos, new_rot, out_rows, is_root=False)
            else:
                flat_row = [
                    1,
                    color,
                    round(new_pos[0], 2),
                    round(new_pos[1], 2),
                    round(new_pos[2], 2),
                    round(new_rot[0][0], 4), round(new_rot[0][1], 4), round(new_rot[0][2], 4),
                    round(new_rot[1][0], 4), round(new_rot[1][1], 4), round(new_rot[1][2], 4),
                    round(new_rot[2][0], 4), round(new_rot[2][1], 4), round(new_rot[2][2], 4),
                    ref
                ]
                out_rows.append(flat_row)

        # ---- GROUP END (if needed) ----
        if wrap_in_group:
            out_rows.append("0 !LEOCAD GROUP END")


    # Exports one specific submodel, including optional nested submodels:
    def ExportFlattenedSubModel(self, submodel_name, out_file):
        """
        Export one submodel fully flattened to a single LDR
        """
        if submodel_name not in self._SubModels:
            raise ValueError(f"Submodel '{submodel_name}' not found")

        out_rows = []
        identity_rot = [
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ]
        origin = [0,0,0]

        self._flatten_submodel(submodel_name, origin, identity_rot, out_rows, is_root=True)

        with open(out_file, "w") as f:
            f.write(f"0 Name: {submodel_name} (flattened)\n")
            for r in out_rows:
                if isinstance(r, list):
                    f.write(" ".join(str(x) for x in r) + "\n")
                else:
                    f.write(r + "\n")

        print(f"> Flattened submodel exported: {out_file}")


# ---------- Show/Export info ----------
    def ExportBOM(self, exportfile=None):
        #Export sorted BOM of all seperate models:
        # 1a. Create header for all subModels (=Key from all subModels)
        # 1b. Create one list of all unique parts used, in all subModels (=Keys from model BOM)
        _Header = ["BOM:"]
        _NumberOfParts = ["NrOfParts:"]
        _NumberOfSteps = ["NrOfSteps:"]
        _UniquePartNames = []
        for model in self._SubModels:
            _Header.append(model)     
            _NumberOfParts.append( self._SubModels[model]._NumberOfParts )
            _NumberOfSteps.append( self._SubModels[model]._NrOfSteps+1 )            
            self._SubModels[model].CreateBOM()
            for key in self._SubModels[model]._BOM.keys():
                if key not in _UniquePartNames:
                    _UniquePartNames.append(key)
        # 2. Sort this unique parts list
        _UniquePartNames.sort()
        # 3a. Export header.
        if exportfile==None:            
            print (_Header)
            print (_NumberOfParts)
            print (_NumberOfSteps)
        if exportfile=="table":
            t = PrettyTable(_Header)
            t.add_row( _NumberOfParts )
            t.add_row( _NumberOfSteps )
        # 3b. Export per list item: amount used per subModel.
        for item in _UniquePartNames:
            _RowBOM = [ item ]            
            for model in self._SubModels:
                _RowBOM.append( self._SubModels[model]._BOM.get(item, 0) )
            if exportfile==None:
                print(_RowBOM)
            else:
                t.add_row( _RowBOM )

        if exportfile=="table":
            print(t)
        print ("Exporting BOM done.")

# ---------- Show model info ----------
    def ShowModelInfo(self):
        print( "> Nr of parts per submodel: ")
        i = 0
        for key in self._SubModels:
            if self.GetKeyMainModel()==key:
                t = PrettyTable()   
                t.field_names = ["subModel","#Obj"] 
                t.align["subModel"] = "l"
                t.add_row( [ f"{i}: {key}", self._SubModels[key]._NumberOfParts] )
            else:
                t.add_row( [ f"{i}: {key}", self._SubModels[key]._NumberOfParts] )
            i += 1
        if i>0:
            print(t)
        else:
            print("> No model data available.")
            
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    my_models = LDrawModel()
    
    # LOAD & UPDATE LDR / MPD (remove duplicates etc):
    #my_models.LoadLDrawModel( r"D:/Github/LDraw/_MODELS/org/more/fabriek_submodellen.mpd" )
    #my_models.LoadLDrawModel( r"D:/Github/LDraw/_MODELS/_NewMPDmodel.mpd" )
    #my_models.LoadLDrawModel( r"D:/Github/LDraw/_MODELS/_New-LDR-as-MPD.ldr" )
    #my_models.LoadLDrawModel( r"D:/Github/LDraw/_MODELS/_TestDecals.ldr" )
    #my_models.LoadLDrawModel( r"D:/Github/LDraw/_MODELS/ORG/ledifice_MODELE_TOURS_(TR)_v1.5.mpd" )
    #my_models.LoadLDrawModel( r"D:/Github/LDraw/_MODELS/ORG/mobaco_User-CasparMol_v1.5.mpd" )
    #my_models.LoadLDrawModel( r"D:/Github/LDraw/_MODELS/test_submodels.mpd" )
    my_models.LoadLDrawModel( r"D:/Github/LDraw/_MODELS/LEdifice_BOXES_0-6_v1.5.mpd" )
    #my_models.LoadDirectory( r"D:/Github/LDraw/_MODELS/EXP/" )


    # Update & save models (partnames & scale):
    #my_parts = parts_db.PartsDB()
    #my_parts.import_csv_library()

    #my_parts.import_csv_library('lib_ConstructionSystemsLEdifice_All_EXP.csv')
    #my_models.UpdateLDrawfiles(r"D:/Github/LDraw/_MODELS/ORG/", ["TATAMI.mpd"],"","_v1.5", 2, my_parts ) 
    #my_models.UpdateLDrawfiles(r"D:/Github/LDraw/_MODELS/import/", ["Jumbo Notre Dame v3i-met zijdeuren.ldr","Moubal fabriek klein-v5-CM.ldr"],"","_v1.5", 2*60/40, my_parts ) #SCALE ONLY: 1, 2 or 60/40
    #my_models.UpdateLDrawfiles(r"D:/Github/LDraw/_MODELS/import/", ["Stadhuis Dordrecht v3.ldr"],"","_v1.5", 1, my_parts ) #SCALE ONLY: 1, 2 or 60/40
    #my_models.UpdateLDrawfiles(r"D:/Github/LDraw/_MODELS/ORG/", ["ELBA_v1.2.mpd"],"","_v1.5", 1, my_parts ) #SCALE ONLY: 1, 2 or 60/40
    #my_models.UpdateLDrawfiles(r"D:/Github/LDraw/_MODELS/ORG/", ["LEdifice_SEREIES-PANELS.ldr",'LEdifice_UNIVERSE_v11.mpd'],"","_v1.5", 1, my_parts ) #SCALE ONLY: 1, 2 or 60/40
    #my_models.UpdateLDrawfiles(r"D:/Github/LDraw/_MODELS/UPDATED/", ["ledifice_Modele_Tours_v1.5.mpd", "LEdifice_SEREIES-PANELS_v1.5.ldr",'LEdifice_UNIVERSE_v1.5.mpd'],"","_UPDT", 1, my_parts ) #NO SCALE.
    #my_models.UpdateLDrawfiles(r"D:/Github/LDraw/_MODELS/ORG/", ["JEUJURA_Universe_v1.1.mpd"],"","_v1.5", 1.25, my_parts ) #SCALE ONLY: 1, 2 or 60/40

    # Model info:
    my_models.ShowModelInfo()
    #my_models.ExportBOM(None) # None, "table"

    # Save models to directory:
    #my_models.SaveLDrawModel( r"D:/Github/LDraw/_MODELS/ORG/Profilo_v1.5.mpd" )
    #my_models.SaveLDrawModel( r"D:/Github/LDraw/_MODELS/ORG/LEdifice_SPECIAL_MODELS_(HS)_v1.5_clean.mpd" )
    #my_models.ExportMPDtoLDR( r"D:/Github/LDraw/_MODELS/EXP/", "", "_single_",True, True )


    # Flatten submodel:
    my_models.ExportFlattenedSubModel( 'LEdifice_Box_0.mpd', 'D:/Github/LDraw/_MODELS/LEdifice_Box_0_FLAT.ldr' )

    #Rename files - rename criteria still in function:
    ##RenameFiles(r"D:/NC-data/ConstructionSys_Vintage/L'EDIFICE/SCANS_TEXTURES/TEXTURES/TEST_JPG/")