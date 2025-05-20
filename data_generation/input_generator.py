import numpy as np
import os
import pandas as pd
import sys

# Get file paths and content
skin_params_file = 'Expanded In Vivo Parameters.csv'  
master_input_file = 'V8Template.inp' 
csv_file_path = os.path.join(os.getcwd(), skin_params_file)
input_file_path = os.path.join(os.getcwd(), master_input_file)

skin_params = pd.read_csv(csv_file_path, header=None)
labels_skin_params = skin_params.columns

# Convert from raw values to abaqus input parameters
def material_to_input_values(params):
    c10 = params[0] / 2
    d = 0
    k1 = params[1]
    k2 = params[2]
    kappa = params[3]
    alpha = params[4]
    return [c10, d, k1, k2, kappa, alpha]


alpha_params = []

# ==== New Alpha Stuff ==== 
### Set up the structure to sample alpha
# Set alpha_11 to 0.05 and -0.05
alpha_pairs = set()
alpha_scale = np.linspace(-0.05, 0.05, 7)

for element in alpha_scale:
    rounded_element = element.round(9)
    alpha_pairs.add((0.05, rounded_element))
    alpha_pairs.add((-0.05, rounded_element))

for element in alpha_pairs:
    alpha_params.append(list(element))

# ==== End of new Expansion ==== 

# Write the task file
with open('master_task_file.in', 'w') as task_file:
    for alpha_set in range(len(alpha_params)): 
        for skin_set in range(len(skin_params)): 
            current_skin_params = skin_params.iloc[skin_set]
            experiment_name = f'a[{alpha_params[alpha_set][0]},{alpha_params[alpha_set][1]}]s{skin_set}'

            material_skin_params = material_to_input_values(current_skin_params)
            c10 = material_skin_params[0]
            d = material_skin_params[1]
            k1 = material_skin_params[2]
            k2 = material_skin_params[3]
            kappa = material_skin_params[4] 
            orientation = material_skin_params[5]

            a11 = alpha_params[alpha_set][0]
            a22 = alpha_params[alpha_set][1]

            # Create destination files for each experiment
            task_file.write(f'cp {master_input_file} {experiment_name}.inp && ' # Create Input File
                            f'python replace_parameters.py {experiment_name} {c10} {d} {k1} {k2} {kappa} {orientation} {a11} {a22} && ' # Replace Parameters
                            f'mkdir {experiment_name}_files && ' # Create Folder directory
                            f'mv {experiment_name}.inp {experiment_name}_files && ' # Move input file into the new directory
                            f'cp retrieve_strain_values.py {experiment_name}_files && cp data_processor.py {experiment_name}_files && ' # Copy over files to get strain csv files
                            f'cd {experiment_name}_files && ' # Switch Directories as current directory
                            f'abaqus job={experiment_name} input={experiment_name}.inp cpu=6 interactive && ' # Run simulation ----Change CPU depending on nodes needed----
                            f'abaqus python retrieve_strain_values.py {experiment_name} && ' # Get strain values
                            f'python data_processor.py {skin_set} {a11} {a22} && '
                            f'rm retrieve_strain_values.py && rm data_processor.py && rm {experiment_name}.odb' # Remove unneeded files
                            '\n'
                        )