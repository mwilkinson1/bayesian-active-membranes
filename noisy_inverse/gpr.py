import pandas as pd
import pickle
import sys
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection import train_test_split

# Check command line arguments
if len(sys.argv) != 2:
    print("Usage: python gpr.py <noise_type>")
    print("Example: python gpr.py s1")
    print("noise_type must be one of: s1, s5, s10, m1, m5, m10, l1, l5, l10")
    sys.exit(1)

noise_type = sys.argv[1]

# Validate noise type
valid_types = ['s1', 's5', 's10', 'm1', 'm5', 'm10', 'l1', 'l5', 'l10']
if noise_type not in valid_types:
    print(f"Error: noise_type must be one of: {valid_types}")
    sys.exit(1)

# Define noise levels 
noise_levels = {
    's1': 1e-1, 's5': 1e0, 's10': 1e0,
    'm1': 1e0, 'm5': 5e0, 'm10': 1e1,
    'l1': 1e1, 'l5': 1e2, 'l10': 1e2
}

noise = noise_levels[noise_type]
print(f"\nTraining models for {noise_type} with noise level {noise}")

# Read the data
try:
    LE_XX_data = pd.read_csv(f"LE_XX_comps_train_{noise_type}.csv", header=None, index_col=0)
    LE_YY_data = pd.read_csv(f"LE_YY_comps_train_{noise_type}.csv", header=None, index_col=0)
    labels = pd.read_csv("param_train.csv", header=None, index_col=0)
except FileNotFoundError as e:
    print(f"Error: Could not find data files: {e}")
    sys.exit(1)

# Split the dataset into train and validation sets
split_ratio = 0.9
GP_train_labels, GP_test_labels = train_test_split(labels, test_size=split_ratio, random_state=123)
XX_train_targets, XX_test_targets = train_test_split(LE_XX_data, test_size=split_ratio, random_state=123)
YY_train_targets, YY_test_targets = train_test_split(LE_YY_data, test_size=split_ratio, random_state=123)

# Define kernel function and GPs
rbf = ConstantKernel(1) * RBF(length_scale=[0.001, 0.01, 1, 0.1, 10, 0.01, 0.01])
gpr_YY = GaussianProcessRegressor(kernel=rbf, alpha=noise)
gpr_XX = GaussianProcessRegressor(kernel=rbf, alpha=noise)

# Fit the models
gpr_YY.fit(GP_train_labels, YY_train_targets)
gpr_XX.fit(GP_train_labels, XX_train_targets)

# Evaluate the models
predict_score_YY = gpr_YY.score(GP_train_labels, YY_train_targets)
print(f'YY Training score: {predict_score_YY}')

test_score_YY = gpr_YY.score(GP_test_labels, YY_test_targets)
print(f'YY Test score: {test_score_YY}')

predict_score_XX = gpr_XX.score(GP_train_labels, XX_train_targets)
print(f'XX Training score: {predict_score_XX}')

test_score_XX = gpr_XX.score(GP_test_labels, XX_test_targets)
print(f'XX Test score: {test_score_XX}')

print(f'YY Kernel: {gpr_YY.kernel_.get_params()}')
print(f'XX Kernel: {gpr_XX.kernel_.get_params()}')

# Save the models
with open(f'XX_GP_{noise_type}.pkl', 'wb') as f:
    pickle.dump(gpr_XX, f)

with open(f'YY_GP_{noise_type}.pkl', 'wb') as f:
    pickle.dump(gpr_YY, f)
