# This code splits up the strain data for the multi-observation cases for the inverse problem
import numpy as np
import pandas as pd
import torch
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python data_spliter.py <noise_data>")
    sys.exit(1)

noise_data = sys.argv[1]

# Create output directory
output_dir = f"{noise_data}_partitioned_data"
os.makedirs(output_dir, exist_ok=True)

# Load in our data for validation using the noise_data variable
labels = pd.read_csv("param_test.csv", header=None, index_col=0)
xx_targets = pd.read_csv(f"LE_XX_comps_test_{noise_data}.csv", header=None, index_col=0)
yy_targets = pd.read_csv(f"LE_YY_comps_test_{noise_data}.csv", header=None, index_col=0)

labels_tensor = torch.tensor(labels.values)
xx_targets_tensor = torch.tensor(xx_targets.values)
yy_targets_tensor = torch.tensor(yy_targets.values)

# We first compare against the 3 design scenaros above and see
def retrieve_t_samples(n_samples, skin_sample, expansion_sample):
    # Define the number of elements to select
    n = n_samples # Change this to the desired number of elements

    # Generate the index list
    indices = [i for i in range(len(labels)) if (i % 10) >= (10 - n)]
    last_frames = labels.iloc[indices]
    j = skin_sample # Skin sample selection
    k = expansion_sample # Expansion selection (9, 12, 13)
    l = 14 * j + k
    selection = last_frames[n * l: n * (l+1)]
    return selection

### 2) Different label split
def retrieve_expansion_samples(skin_sample):
    return labels[9::10][14 * skin_sample: 14 * (skin_sample + 1)]

uni_list = []
max_list = []
mixed_list = []
skin_sample_selection = [1, 2, 18]

for s in skin_sample_selection:
    uni_list.append(retrieve_t_samples(10, s, 13))
    max_list.append(retrieve_t_samples(10, s, 12))
    mixed_list.append(retrieve_t_samples(10, s, 9))

uni_df = pd.concat(uni_list)
max_df = pd.concat(max_list)
mixed_df = pd.concat(mixed_list)

# Save DataFrames to CSV in the output directory
uni_df.to_csv(os.path.join(output_dir, f"{noise_data}_uni_multi_labels.csv"), header=False)
xx_targets.loc[uni_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_xx_uni_multi.csv"), header=False)
yy_targets.loc[uni_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_yy_uni_multi.csv"), header=False)

max_df.to_csv(os.path.join(output_dir, f"{noise_data}_max_multi_labels.csv"), header=False)
xx_targets.loc[max_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_xx_max_multi.csv"), header=False)
yy_targets.loc[max_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_yy_max_multi.csv"), header=False)

mixed_df.to_csv(os.path.join(output_dir, f"{noise_data}_mixed_multi_labels.csv"), header=False)
xx_targets.loc[mixed_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_xx_mixed_multi.csv"), header=False)
yy_targets.loc[mixed_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_yy_mixed_multi.csv"), header=False)

# Set up the 3 membrane expansion scenarios for single sample
# 1) 0.05 and 0.05
max_expansion_labels = labels[(labels.iloc[:,5] == 0.05) * (labels.iloc[:,6] == 0.05)]
max_expansion_xx = xx_targets.loc[max_expansion_labels.index]
max_expansion_yy = yy_targets.loc[max_expansion_labels.index]

# 2) 0.05 and -0.05
mixed_expansion_labels = labels.loc[(labels.iloc[:,5] == 0.05) * (labels.iloc[:,6] == -0.05)]
mixed_expansion_xx = xx_targets.loc[mixed_expansion_labels.index]
mixed_expansion_yy = yy_targets.loc[mixed_expansion_labels.index]

# 3) 0.05 and 0
uniaxial_expansion_labels = labels.loc[(labels.iloc[:,5] == 0.05) * (labels.iloc[:,6] == 0)]
uniaxial_expansion_xx = xx_targets.loc[uniaxial_expansion_labels.index]
uniaxial_expansion_yy = yy_targets.loc[uniaxial_expansion_labels.index]

# Save all DataFrames to CSV in the output directory
max_expansion_labels.to_csv(os.path.join(output_dir, f"{noise_data}_max_expansion_labels.csv"), header=False)
max_expansion_xx.to_csv(os.path.join(output_dir, f"{noise_data}_max_expansion_xx.csv"), header=False)
max_expansion_yy.to_csv(os.path.join(output_dir, f"{noise_data}_max_expansion_yy.csv"), header=False)

mixed_expansion_labels.to_csv(os.path.join(output_dir, f"{noise_data}_mixed_expansion_labels.csv"), header=False)
mixed_expansion_xx.to_csv(os.path.join(output_dir, f"{noise_data}_mixed_expansion_xx.csv"), header=False)
mixed_expansion_yy.to_csv(os.path.join(output_dir, f"{noise_data}_mixed_expansion_yy.csv"), header=False)

uniaxial_expansion_labels.to_csv(os.path.join(output_dir, f"{noise_data}_uniaxial_expansion_labels.csv"), header=False)
uniaxial_expansion_xx.to_csv(os.path.join(output_dir, f"{noise_data}_uniaxial_expansion_xx.csv"), header=False)
uniaxial_expansion_yy.to_csv(os.path.join(output_dir, f"{noise_data}_uniaxial_expansion_yy.csv"), header=False)

# Extract expansion based samples
expansion_labels_1 = retrieve_expansion_samples(1)
expansion_labels_2 = retrieve_expansion_samples(2)
expansion_labels_3 = retrieve_expansion_samples(18)

# Find corresponding PC weights
expansion_xx_1 = xx_targets.loc[expansion_labels_1.index]
expansion_xx_2 = xx_targets.loc[expansion_labels_2.index]
expansion_xx_3 = xx_targets.loc[expansion_labels_3.index]

expansion_yy_1 = yy_targets.loc[expansion_labels_1.index]
expansion_yy_2 = yy_targets.loc[expansion_labels_2.index]
expansion_yy_3 = yy_targets.loc[expansion_labels_3.index]

# T samples
uni_list = []
max_list = []
mixed_list = []
skin_sample_selection = [1, 2, 18]

for s in skin_sample_selection:
    uni_list.append(retrieve_t_samples(10, s, 13))
    max_list.append(retrieve_t_samples(10, s, 12))
    mixed_list.append(retrieve_t_samples(10, s, 9))

uni_df = pd.concat(uni_list)
max_df = pd.concat(max_list)
mixed_df = pd.concat(mixed_list)

# Save DataFrames to CSV in the output directory
uni_df.to_csv(os.path.join(output_dir, f"{noise_data}_uni_multi_labels.csv"), header=False)
xx_targets.loc[uni_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_xx_uni_multi.csv"), header=False)
yy_targets.loc[uni_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_yy_uni_multi.csv"), header=False)

max_df.to_csv(os.path.join(output_dir, f"{noise_data}_max_multi_labels.csv"), header=False)
xx_targets.loc[max_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_xx_max_multi.csv"), header=False)
yy_targets.loc[max_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_yy_max_multi.csv"), header=False)

mixed_df.to_csv(os.path.join(output_dir, f"{noise_data}_mixed_multi_labels.csv"), header=False)
xx_targets.loc[mixed_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_xx_mixed_multi.csv"), header=False)
yy_targets.loc[mixed_df.index].to_csv(os.path.join(output_dir, f"{noise_data}_yy_mixed_multi.csv"), header=False)

expansion_labels_1.to_csv(os.path.join(output_dir, f"{noise_data}_expansion_labels_1.csv"), header=False)
expansion_xx_1.to_csv(os.path.join(output_dir, f"{noise_data}_expansion_xx_1.csv"), header=None)
expansion_yy_1.to_csv(os.path.join(output_dir, f"{noise_data}_expansion_yy_1.csv"), header=None)

expansion_labels_2.to_csv(os.path.join(output_dir, f"{noise_data}_expansion_labels_2.csv"), header=False)
expansion_xx_2.to_csv(os.path.join(output_dir, f"{noise_data}_expansion_xx_2.csv"), header=None)
expansion_yy_2.to_csv(os.path.join(output_dir, f"{noise_data}_expansion_yy_2.csv"), header=None)

expansion_labels_3.to_csv(os.path.join(output_dir, f"{noise_data}_expansion_labels_3.csv"), header=False)
expansion_xx_3.to_csv(os.path.join(output_dir, f"{noise_data}_expansion_xx_3.csv"), header=None)
expansion_yy_3.to_csv(os.path.join(output_dir, f"{noise_data}_expansion_yy_3.csv"), header=None)
