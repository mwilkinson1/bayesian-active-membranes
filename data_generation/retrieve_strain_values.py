import sys
import csv
import numpy as np
from odbAccess import openOdb
from abaqusConstants import *

current_odb = str(sys.argv[-1])

# Abaqus Scripting useful commands
odb = openOdb('{file}.odb'.format(file = current_odb))
skin_elements = odb.rootAssembly.elementSets['SET-4']

inputFrames = odb.steps['Step-1'].frames

for index, item in enumerate(inputFrames):
    if index == 0:
        continue # Get rid of that first one with no stress
  
    # Log strain
    Lstrain = inputFrames[index].fieldOutputs['LE']
    Lstrain_top = Lstrain.getSubset(region=skin_elements).values

    LE_csv_path = open('LEt{i}.csv'.format(i = index), 'w')
    c_LE = csv.writer(LE_csv_path)
    # c_LE.writerow(['Element', 'Integration Point', ['LE11', 'LE22', 'LE33', 'LE12', 'LE13', 'LE23']])

    for i, element in enumerate(Lstrain_top):
        LE_entry = [Lstrain_top[i].elementLabel, Lstrain_top[i].integrationPoint , Lstrain_top[i].data]
        c_LE.writerow(LE_entry)

    LE_csv_path.close()

    NE_csv_path = open('NEt{i}.csv'.format(i = index), 'w')
    c_NE = csv.writer(NE_csv_path)
    # c_NE.writerow(['Element', 'Integration Point', ['NE11', 'NE22', 'NE33', 'NE12', 'NE13', 'NE23']])
        
    # Nominal Strain
    Nstrain = inputFrames[index].fieldOutputs['NE']
    Nstrain_top = Nstrain.getSubset(region=skin_elements).values

    for i, element in enumerate(Nstrain_top):
        NE_entry = [Nstrain_top[i].elementLabel, Nstrain_top[i].integrationPoint , Nstrain_top[i].data]
        c_NE.writerow(NE_entry)
    NE_csv_path.close()
    