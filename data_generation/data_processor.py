import pandas as pd
import numpy as np
import os
import sys

skin_param_set = int(sys.argv[1])
a11 = float(sys.argv[2])
a22 = float(sys.argv[3])

# Include the alpha expansions when change parameters
alpha_params = []

# Set alpha_11 to 0.05 and -0.05
alpha_pairs = set()
alpha_scale = np.linspace(-0.05, 0.05, 7)

for element in alpha_scale:
    rounded_element = element.round(9)
    alpha_pairs.add((0.05, rounded_element))
    alpha_pairs.add((-0.05, rounded_element))

for element in alpha_pairs:
    alpha_params.append(list(element))

# Change this depending on what skin parameters you want to change
filename_skin_params = 'Expanded In Vivo Parameters.csv'

def string_to_list(input_string):
    cleaned_string = input_string.strip('[]')
    numbers_str = cleaned_string.split()
    numbers_list = [float(num) for num in numbers_str]

    return numbers_list

def element_wise_average(lists):
    num_lists = len(lists)
    num_elements = len(lists[0])
    result = [sum(lists[i][j] for i in range(num_lists)) / num_lists for j in range(num_elements)]
    return result

def save_to_master_file(input_strain_data, strain_type, t, skin_set, a1, a2):
    # Get the current working directory
    current_directory = os.getcwd()

    # Navigate to the parent directory
    parent_directory = os.path.dirname(current_directory)

    # Also get the skin params file
    skin_csv_path = os.path.join(parent_directory, filename_skin_params)
    skin_params = pd.read_csv(skin_csv_path, header=None)

    current_skin_params = list(skin_params.iloc[skin_set])
    
    parameter_set_file_path = os.path.join(current_directory, 'Parameters.csv')

    # Construct the full path to the CSV file in the parent directory
    if strain_type == 'NE':
        csv_file_path = os.path.join(current_directory, 'NE Strain.csv')

    if strain_type == 'LE':
        csv_file_path = os.path.join(current_directory, 'LE Strain.csv')

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        # If it doesn't exist, create a new DataFrame
        df = pd.DataFrame(columns=[f's={skin_param_set}, a11={round(a11 * t, 7)}, a22={round(a22 * t, 7)}'])  # You can modify column names as per your requirements
    else:
        # If it exists, read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
    
    # Do the same but for the parameter file
        
    if not os.path.exists(parameter_set_file_path):
        # If it doesn't exist, create a new DataFrame
        df_params = pd.DataFrame(columns=[f's={skin_param_set}, a11={round(a11 * t, 7)}, a22={round(a22 * t, 7)}'])  # You can modify column names as per your requirements
    else:
        # If it exists, read the CSV file into a DataFrame
        df_params = pd.read_csv(parameter_set_file_path)

    alpha_set = list([a1 * t, a2 * t])
    # Add your new column of data to the DataFrame
    new_column_name = f's={skin_set}, a11={round(a11 * t, 7)}, a22={round(a22 * t, 7)}'
    df[new_column_name] = input_strain_data
    df_params[new_column_name] = current_skin_params[0:5] + alpha_set

    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)
    df_params.to_csv(parameter_set_file_path, index=False)


for i in range(1, 11): 

    LE_file_name = f"LEt{i}.csv"
    NE_file_name = f"NEt{i}.csv"

    LE_file_path = os.path.join(os.getcwd(), LE_file_name)
    NE_file_path = os.path.join(os.getcwd(), NE_file_name)
                                
    NE_frame = pd.read_csv(NE_file_path, header=None)
    LE_frame = pd.read_csv(LE_file_path, header=None)

    LE_strain_elements = LE_frame[2] # Strain column
    NE_strain_elements = NE_frame[2] # Strain column

    # Cleaning steps
    cleaned_LE_strain_elements = []
    cleaned_NE_strain_elements = []

    for element_strain in LE_strain_elements:
        cleaned_LE_vals = string_to_list(element_strain)
        cleaned_LE_strain_elements.append(cleaned_LE_vals)

    for element_strain in NE_strain_elements:
        cleaned_NE_vals = string_to_list(element_strain)
        cleaned_NE_strain_elements.append(cleaned_NE_vals)
        
    LE_frame.iloc[:,2] = pd.Series(cleaned_LE_strain_elements)
    NE_frame.iloc[:,2] = pd.Series(cleaned_NE_strain_elements)

    L_top_int_points = LE_frame[LE_frame[1].isin([1,2,3,4])]
    N_top_int_points = NE_frame[NE_frame[1].isin([1,2,3,4])]

    LE_averaged_strain = L_top_int_points.groupby(L_top_int_points.index // 4)[2].apply(lambda x: element_wise_average(x.tolist())).reset_index(drop=True)
    NE_averaged_strain = N_top_int_points.groupby(N_top_int_points.index // 4)[2].apply(lambda x: element_wise_average(x.tolist())).reset_index(drop=True)

    # Change so that there is only the xx, zz, and xz components

    LE_surface_components = []
    NE_surface_components = []

    for j in range(len(LE_averaged_strain)):
        LE_surface_components.append([LE_averaged_strain[j][0], LE_averaged_strain[j][2], LE_averaged_strain[j][4]])
        NE_surface_components.append([NE_averaged_strain[j][0], NE_averaged_strain[j][2], NE_averaged_strain[j][4]])

    LE_surface_components = pd.Series(LE_surface_components)
    NE_surface_components = pd.Series(NE_surface_components)

    # Finally save these 3 components to the final file
    save_to_master_file(NE_surface_components, 'NE', i/10, skin_param_set, a11, a22)
    save_to_master_file(LE_surface_components, 'LE', i/10, skin_param_set, a11, a22)

    # Remove strain files
    os.remove(f'NEt{i}.csv')
    os.remove(f'LEt{i}.csv')
    

