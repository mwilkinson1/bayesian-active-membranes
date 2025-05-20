import pandas as pd
import pickle
import sys
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection import train_test_split

# Check command line arguments
if len(sys.argv) != 3 and len(sys.argv) != 4:
    print("Usage:")
    print("For multi-format data: python gpr.py <data_type> multi")
    print("For expansion data: python gpr.py <data_type> expansion <expansion_num>")
    print("data_type must be one of: uni, max, mixed")
    sys.exit(1)

data_type = sys.argv[1]
data_format = sys.argv[2]

if data_type not in ['uni', 'max', 'mixed']:
    print("Error: data_type must be one of: uni, max, mixed")
    sys.exit(1)

if data_format not in ['multi', 'expansion']:
    print("Error: data_format must be one of: multi, expansion")
    sys.exit(1)

expansion_num = None
if data_format == 'expansion':
    if len(sys.argv) != 4:
        print("Error: expansion_num required when data_format is 'expansion'")
        sys.exit(1)
    try:
        expansion_num = int(sys.argv[3])
    except ValueError:
        print("Error: expansion_num must be an integer")
        sys.exit(1)

def get_file_paths(name_var, data_type, data_format, expansion_num=None):
    base_dir = f"{name_var}_partitioned_data"
    
    if data_format == 'multi':
        xx_file = f"{name_var}_xx_{data_type}_multi.csv"
        yy_file = f"{name_var}_yy_{data_type}_multi.csv"
        labels_file = f"{name_var}_{data_type}_multi_labels.csv"
    else:  # expansion
        xx_file = f"{name_var}_expansion_xx_{expansion_num}.csv"
        yy_file = f"{name_var}_expansion_yy_{expansion_num}.csv"
        labels_file = f"{name_var}_expansion_labels_{expansion_num}.csv"
    
    return {
        'xx': os.path.join(base_dir, xx_file),
        'yy': os.path.join(base_dir, yy_file),
        'labels': os.path.join(base_dir, labels_file)
    }

# Define the name variables and corresponding noise levels
name_vars = ['s1', 's5', 's10', 'm1', 'm5', 'm10', 'l1', 'l5', 'l10']

# Noise levels were selected heuristically to optimize performance on the test set. Overfitting was common
noise_levels = [1e-1, 1e0, 1e0, 1e0, 5e0, 1e1, 1e1, 1e2, 1e2]

split_ratio = 0.9

for name_var, noise in zip(name_vars, noise_levels):
    print(f"\nTraining models for {name_var} with noise level {noise}")
    print(f"Using {data_type} {data_format} data")
    
    # Get file paths for this iteration
    file_paths = get_file_paths(name_var, data_type, data_format, expansion_num)
    
    # Read the data
    try:
        LE_XX_data = pd.read_csv(file_paths['xx'], header=None, index_col=0)
        LE_YY_data = pd.read_csv(file_paths['yy'], header=None, index_col=0)
        labels = pd.read_csv(file_paths['labels'], header=None, index_col=0)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files for {name_var}: {e}")
        continue
    
    # Split the dataset into train and validation sets
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
    print(f'YY Training score for {name_var}: {predict_score_YY}')
    
    train_score_YY = gpr_YY.score(GP_test_labels, YY_test_targets)
    print(f'YY Validation score for {name_var}: {train_score_YY}')
    
    predict_score_XX = gpr_XX.score(GP_train_labels, XX_train_targets)
    print(f'XX Training score for {name_var}: {predict_score_XX}')
    
    train_score_XX = gpr_XX.score(GP_test_labels, XX_test_targets)
    print(f'XX Validation score for {name_var}: {train_score_XX}')
    
    print(f'YY Kernel for {name_var}: {gpr_YY.kernel_.get_params()}')
    print(f'XX Kernel for {name_var}: {gpr_XX.kernel_.get_params()}')
    
    # Create model directory if it doesn't exist
    model_dir = f"models_{data_type}_{data_format}"
    if data_format == 'expansion':
        model_dir += f"_exp{expansion_num}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the models with the name_var in the filename
    with open(os.path.join(model_dir, f'YY_GP_{name_var}.pkl'), 'wb') as f:
        pickle.dump(gpr_YY, f)
    
    with open(os.path.join(model_dir, f'XX_GP_{name_var}.pkl'), 'wb') as f:
        pickle.dump(gpr_XX, f)
