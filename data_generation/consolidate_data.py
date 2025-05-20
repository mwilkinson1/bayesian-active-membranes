import os
import pandas as pd

# Get the list of all files and folders in the current directory
files_and_folders = os.listdir()

# Filter out only the folders
folders = [folder for folder in files_and_folders if os.path.isdir(folder)]
folders.sort()

NE_filename = "NE Strain.csv"
LE_filename = "LE Strain.csv"
parameter_filename = "Parameters.csv"
NE_df = pd.DataFrame([])
LE_df = pd.DataFrame([])
parameter_df = pd.DataFrame([])

for f in folders:
    folder_contents = os.listdir(f)
    NE_filepath = os.path.join(f, NE_filename)
    LE_filepath = os.path.join(f, LE_filename)
    parameter_filepath = os.path.join(f, parameter_filename)

    f_NE = pd.read_csv(NE_filepath, index_col=None)
    f_LE = pd.read_csv(LE_filepath, index_col=None)
    f_parameters = pd.read_csv(parameter_filepath, index_col=None)
    
    NE_df = pd.concat([NE_df, f_NE], axis=1)
    LE_df = pd.concat([LE_df, f_LE], axis=1)
    parameter_df = pd.concat([parameter_df, f_parameters], axis=1)

NE_df.to_csv("NE Strain Combined.csv", index=False, header=False)
LE_df.to_csv("LE Strain Combined.csv", index=False, header=False)
parameter_df.to_csv("Parameters Combined.csv", index=False, header=False)