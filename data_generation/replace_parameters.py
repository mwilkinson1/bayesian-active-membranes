import numpy as np
import pandas as pd
import sys

# Script inputs:
experiment_name = sys.argv[1]
c10 = sys.argv[2]
d = sys.argv[3]
k1 = sys.argv[4]
k2 = sys.argv[5]
kappa = sys.argv[6]
alpha = sys.argv[7]
a11 = sys.argv[8]
a22 = sys.argv[9]

skin_parameter_line = f'{c10}, {d}, {k1}, {k2}, {kappa}'
expansion_parameter_line = f'{a11}, {a22}, 0'
orientation_parameter_line = f'2, {alpha}'

material_call_line = '*Anisotropic Hyperelastic'
orientation_call_line = '*Orientation, name=Ori-1'
expansion_call_line = '*Expansion, type=ORTHO'

with open(f'{experiment_name}.inp', 'r') as file:
    content = file.readlines()
    for ind, line in enumerate(content):
        if material_call_line in line:
            skin_material_index = ind + 1
            content[skin_material_index] = skin_parameter_line + '\n'
        if expansion_call_line in line:
            expansion_index = ind + 1
            content[expansion_index] = expansion_parameter_line + '\n'
        if orientation_call_line in line:
            orientation_index = ind + 2
            content[orientation_index] = orientation_parameter_line + '\n'

output_string = ''

for element in content:
    output_string = output_string + element

with open(f'{experiment_name}.inp', 'w') as file:
    file.write(output_string)
    file.close()
            