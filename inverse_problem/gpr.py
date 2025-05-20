import pandas as pd
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.model_selection import train_test_split

LE_XX_train_targets = pd.read_csv("LE_XX_comps_train.csv", header=None, index_col=0)
LE_XX_test_targets = pd.read_csv("LE_XX_comps_test.csv", header=None, index_col=0)
LE_YY_train_targets = pd.read_csv("LE_YY_comps_train.csv", header=None, index_col=0)
LE_YY_test_targets = pd.read_csv("LE_YY_comps_test.csv", header=None, index_col=0)
train_labels = pd.read_csv("param_train.csv", header=None, index_col=0)
test_labels = pd.read_csv("param_test.csv", header=None, index_col=0)

split_ratio = 0.9

# Split the dataset into train and validation sets
GP_train_labels, GP_test_labels = train_test_split(train_labels, test_size=split_ratio, random_state=123)

XX_train_targets, XX_test_targets = train_test_split(LE_XX_train_targets, test_size=split_ratio, random_state=123)

YY_train_targets, YY_test_targets = train_test_split(LE_YY_train_targets, test_size=split_ratio, random_state=123)

# For different data sets (i.e. s1, m5, l10, etc.), this will need to be changed
noise = 1e-3

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

train_score_YY = gpr_YY.score(GP_test_labels, YY_test_targets)
print(f'YY Validation score: {train_score_YY}')

predict_score_XX = gpr_XX.score(GP_train_labels, XX_train_targets)
print(f'XX Training score: {predict_score_XX}')

train_score_XX = gpr_XX.score(GP_test_labels, XX_test_targets)
print(f'XX Validation score: {train_score_XX}')

# Compare against our true dataset and see how it performs
test_score_YY = gpr_YY.score(test_labels, LE_YY_test_targets)
print(f'YY Test score: {test_score_YY}')

test_score_XX = gpr_XX.score(test_labels, LE_XX_test_targets)
print(f'XX Test score: {test_score_XX}')

print(f'YY Kernel: {gpr_YY.kernel_.get_params()}')
print(f'XX Kernel: {gpr_XX.kernel_.get_params()}')

# Save the models
with open('YY_GP.pkl', 'wb') as f:
    pickle.dump(gpr_YY, f)

with open('XX_GP.pkl', 'wb') as f:
    pickle.dump(gpr_XX, f)
