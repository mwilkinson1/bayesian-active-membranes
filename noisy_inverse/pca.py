import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read in strain components and parameters
parameter_df = pd.read_csv("Parameters.csv", header=None)

# Load in noisy data
LE_XX_s1 = pd.read_csv("LE_XX_1s.csv", header=None, index_col=None)
LE_XX_m1 = pd.read_csv("LE_XX_1m.csv", header=None, index_col=None)
LE_XX_l1 = pd.read_csv("LE_XX_1l.csv", header=None, index_col=None)

LE_YY_s1 = pd.read_csv("LE_YY_1s.csv", header=None, index_col=None)
LE_YY_m1 = pd.read_csv("LE_YY_1m.csv", header=None, index_col=None)
LE_YY_l1 = pd.read_csv("LE_YY_1l.csv", header=None, index_col=None)

LE_XX_s5 = pd.read_csv("LE_XX_5s.csv", header=None, index_col=None)
LE_XX_m5 = pd.read_csv("LE_XX_5m.csv", header=None, index_col=None)
LE_XX_l5 = pd.read_csv("LE_XX_5l.csv", header=None, index_col=None)

LE_YY_s5 = pd.read_csv("LE_YY_5s.csv", header=None, index_col=None)
LE_YY_m5 = pd.read_csv("LE_YY_5m.csv", header=None, index_col=None)
LE_YY_l5 = pd.read_csv("LE_YY_5l.csv", header=None, index_col=None)

LE_XX_s10 = pd.read_csv("LE_XX_10s.csv", header=None, index_col=None)
LE_XX_m10 = pd.read_csv("LE_XX_10m.csv", header=None, index_col=None)
LE_XX_l10 = pd.read_csv("LE_XX_10l.csv", header=None, index_col=None)

LE_YY_s10 = pd.read_csv("LE_YY_10s.csv", header=None, index_col=None)
LE_YY_m10 = pd.read_csv("LE_YY_10m.csv", header=None, index_col=None)
LE_YY_l10 = pd.read_csv("LE_YY_10l.csv", header=None, index_col=None)


## Read in a mesh from Abaqus
meshfile = open("V8Template.inp",'r').readlines()
n_lines = len(meshfile)
for i in range(n_lines):
    if meshfile[i][0:5]=="*Node" and '*Node Output' not in meshfile[i] and ('name=SKIN&MUSCLE' in meshfile[i-1]):
        end = 0
        count=0
        node_X_array = []
        node_X_ind = []
        while end<1:
            aux = meshfile[i+count+1].split(',')
            if aux==['CF', ' RF', ' U\n']:
                break
            nodeXi = np.array([float(aux[1]),float(aux[2]),float(aux[3])])
            node_X_array.append(nodeXi)
            node_X_ind.append(int(aux[0]))
            count+=1
            if meshfile[i+count+1][0]=="*":
                end = 1

n_node = count
for i in range(n_lines):
    if meshfile[i][0:5]=="*Elem" and ('type=C3D8H' in meshfile[i]):
        end = 0
        count=0
        elem_array = []
        while end<1:
            aux = meshfile[i+count+1].split(',')
            elemi = np.array([int(aux[1]),int(aux[2]),int(aux[3]),int(aux[4]),int(aux[5]),int(aux[6]),int(aux[7]),int(aux[8])])
            elem_array.append(elemi)
            count+=1
            if meshfile[i+count+1][0]=="*":
                end = 1
        if end==1:
            break

n_elem = count

node_X = np.array(node_X_array)
elements = np.array(elem_array)

random.seed(201)

# Define the range and the sample size
range_end = 120
sample_size = 100

# Generate the sample
skin_train_samples = random.sample(range(range_end), sample_size)
skin_train_samples = sorted(skin_train_samples)

# Lets get the other skin parameters indicies
skin_test_samples = [i for i in range(range_end) if i not in skin_train_samples]
skin_test_samples = sorted(skin_test_samples)

experimentwise_parameters = []

for i in range(int(parameter_df.shape[1] / 10)):
    experimentwise_parameters.append(parameter_df.T[i * 10: (i + 1) * 10])

sorted_parameters = []
for i in range(120):
    sorted_parameters.append(experimentwise_parameters[i::120])

param_train_df = [sorted_parameters[train_ind] for train_ind in skin_train_samples]
param_test_df = [sorted_parameters[test_ind] for test_ind in skin_test_samples]

train_indicies = []
test_indicies = []
for skin_set in param_train_df:
    internal = list(pd.concat(skin_set).index)
    train_indicies += internal

for skin_set in param_test_df:
    internal = list(pd.concat(skin_set).index)
    test_indicies += internal

param_train_df = parameter_df.T.loc[train_indicies]
param_test_df = parameter_df.T.loc[test_indicies]

LE_XX_train_s1 = LE_XX_s1.T.loc[train_indicies]
LE_XX_train_m1 = LE_XX_m1.T.loc[train_indicies]
LE_XX_train_l1 = LE_XX_l1.T.loc[train_indicies]
LE_XX_test_s1 = LE_XX_s1.T.loc[test_indicies]
LE_XX_test_m1 = LE_XX_m1.T.loc[test_indicies]
LE_XX_test_l1 = LE_XX_l1.T.loc[test_indicies]

LE_YY_train_s1 = LE_YY_s1.T.loc[train_indicies]
LE_YY_train_m1 = LE_YY_m1.T.loc[train_indicies]
LE_YY_train_l1 = LE_YY_l1.T.loc[train_indicies]
LE_YY_test_s1 = LE_YY_s1.T.loc[test_indicies]
LE_YY_test_m1 = LE_YY_m1.T.loc[test_indicies]
LE_YY_test_l1 = LE_YY_l1.T.loc[test_indicies]

LE_XX_train_s5 = LE_XX_s5.T.loc[train_indicies]
LE_XX_train_m5 = LE_XX_m5.T.loc[train_indicies]
LE_XX_train_l5 = LE_XX_l5.T.loc[train_indicies]
LE_XX_test_s5 = LE_XX_s5.T.loc[test_indicies]
LE_XX_test_m5 = LE_XX_m5.T.loc[test_indicies]
LE_XX_test_l5 = LE_XX_l5.T.loc[test_indicies]

LE_YY_train_s5 = LE_YY_s5.T.loc[train_indicies]
LE_YY_train_m5 = LE_YY_m5.T.loc[train_indicies]
LE_YY_train_l5 = LE_YY_l5.T.loc[train_indicies]
LE_YY_test_s5 = LE_YY_s5.T.loc[test_indicies]
LE_YY_test_m5 = LE_YY_m5.T.loc[test_indicies]
LE_YY_test_l5 = LE_YY_l5.T.loc[test_indicies]

LE_XX_train_s10 = LE_XX_s10.T.loc[train_indicies]
LE_XX_train_m10 = LE_XX_m10.T.loc[train_indicies]
LE_XX_train_l10 = LE_XX_l10.T.loc[train_indicies]
LE_XX_test_s10 = LE_XX_s10.T.loc[test_indicies]
LE_XX_test_m10 = LE_XX_m10.T.loc[test_indicies]
LE_XX_test_l10 = LE_XX_l10.T.loc[test_indicies]

LE_YY_train_s10 = LE_YY_s10.T.loc[train_indicies]
LE_YY_train_m10 = LE_YY_m10.T.loc[train_indicies]
LE_YY_train_l10 = LE_YY_l10.T.loc[train_indicies]
LE_YY_test_s10 = LE_YY_s10.T.loc[test_indicies]
LE_YY_test_m10 = LE_YY_m10.T.loc[test_indicies]
LE_YY_test_l10 = LE_YY_l10.T.loc[test_indicies]

LE_XX_scalar = preprocessing.StandardScaler()
LE_YY_scalar = preprocessing.StandardScaler()

# Initializing separate scalers for each size and time point
LE_XX_scalar_s1 = preprocessing.StandardScaler()
LE_XX_scalar_m1 = preprocessing.StandardScaler()
LE_XX_scalar_l1 = preprocessing.StandardScaler()

LE_YY_scalar_s1 = preprocessing.StandardScaler()
LE_YY_scalar_m1 = preprocessing.StandardScaler()
LE_YY_scalar_l1 = preprocessing.StandardScaler()

LE_XX_scalar_s5 = preprocessing.StandardScaler()
LE_XX_scalar_m5 = preprocessing.StandardScaler()
LE_XX_scalar_l5 = preprocessing.StandardScaler()

LE_YY_scalar_s5 = preprocessing.StandardScaler()
LE_YY_scalar_m5 = preprocessing.StandardScaler()
LE_YY_scalar_l5 = preprocessing.StandardScaler()

LE_XX_scalar_s10 = preprocessing.StandardScaler()
LE_XX_scalar_m10 = preprocessing.StandardScaler()
LE_XX_scalar_l10 = preprocessing.StandardScaler()

LE_YY_scalar_s10 = preprocessing.StandardScaler()
LE_YY_scalar_m10 = preprocessing.StandardScaler()
LE_YY_scalar_l10 = preprocessing.StandardScaler()

LE_XX_scaled_s1 = LE_XX_scalar_s1.fit_transform(LE_XX_train_s1)
LE_XX_scaled_m1 = LE_XX_scalar_m1.fit_transform(LE_XX_train_m1)
LE_XX_scaled_l1 = LE_XX_scalar_l1.fit_transform(LE_XX_train_l1)

LE_YY_scaled_s1 = LE_YY_scalar_s1.fit_transform(LE_YY_train_s1)
LE_YY_scaled_m1 = LE_YY_scalar_m1.fit_transform(LE_YY_train_m1)
LE_YY_scaled_l1 = LE_YY_scalar_l1.fit_transform(LE_YY_train_l1)

LE_XX_scaled_s5 = LE_XX_scalar_s5.fit_transform(LE_XX_train_s5)
LE_XX_scaled_m5 = LE_XX_scalar_m5.fit_transform(LE_XX_train_m5)
LE_XX_scaled_l5 = LE_XX_scalar_l5.fit_transform(LE_XX_train_l5)

LE_YY_scaled_s5 = LE_YY_scalar_s5.fit_transform(LE_YY_train_s5)
LE_YY_scaled_m5 = LE_YY_scalar_m5.fit_transform(LE_YY_train_m5)
LE_YY_scaled_l5 = LE_YY_scalar_l5.fit_transform(LE_YY_train_l5)

LE_XX_scaled_s10 = LE_XX_scalar_s10.fit_transform(LE_XX_train_s10)
LE_XX_scaled_m10 = LE_XX_scalar_m10.fit_transform(LE_XX_train_m10)
LE_XX_scaled_l10 = LE_XX_scalar_l10.fit_transform(LE_XX_train_l10)

LE_YY_scaled_s10 = LE_YY_scalar_s10.fit_transform(LE_YY_train_s10)
LE_YY_scaled_m10 = LE_YY_scalar_m10.fit_transform(LE_YY_train_m10)
LE_YY_scaled_l10 = LE_YY_scalar_l10.fit_transform(LE_YY_train_l10)

# Transform variables from the test set using the respective scalers
LE_XX_test_scaled_s1 = LE_XX_scalar_s1.transform(LE_XX_test_s1)
LE_XX_test_scaled_m1 = LE_XX_scalar_m1.transform(LE_XX_test_m1)
LE_XX_test_scaled_l1 = LE_XX_scalar_l1.transform(LE_XX_test_l1)

LE_YY_test_scaled_s1 = LE_YY_scalar_s1.transform(LE_YY_test_s1)
LE_YY_test_scaled_m1 = LE_YY_scalar_m1.transform(LE_YY_test_m1)
LE_YY_test_scaled_l1 = LE_YY_scalar_l1.transform(LE_YY_test_l1)

LE_XX_test_scaled_s5 = LE_XX_scalar_s5.transform(LE_XX_test_s5)
LE_XX_test_scaled_m5 = LE_XX_scalar_m5.transform(LE_XX_test_m5)
LE_XX_test_scaled_l5 = LE_XX_scalar_l5.transform(LE_XX_test_l5)

LE_YY_test_scaled_s5 = LE_YY_scalar_s5.transform(LE_YY_test_s5)
LE_YY_test_scaled_m5 = LE_YY_scalar_m5.transform(LE_YY_test_m5)
LE_YY_test_scaled_l5 = LE_YY_scalar_l5.transform(LE_YY_test_l5)

LE_XX_test_scaled_s10 = LE_XX_scalar_s10.transform(LE_XX_test_s10)
LE_XX_test_scaled_m10 = LE_XX_scalar_m10.transform(LE_XX_test_m10)
LE_XX_test_scaled_l10 = LE_XX_scalar_l10.transform(LE_XX_test_l10)

LE_YY_test_scaled_s10 = LE_YY_scalar_s10.transform(LE_YY_test_s10)
LE_YY_test_scaled_m10 = LE_YY_scalar_m10.transform(LE_YY_test_m10)
LE_YY_test_scaled_l10 = LE_YY_scalar_l10.transform(LE_YY_test_l10)


# Initializing PCA for each size and time point
pca_LE_XX_s1 = PCA(n_components=8, svd_solver="randomized")
pca_LE_XX_m1 = PCA(n_components=8, svd_solver="randomized")
pca_LE_XX_l1 = PCA(n_components=8, svd_solver="randomized")

pca_LE_YY_s1 = PCA(n_components=8, svd_solver="randomized")
pca_LE_YY_m1 = PCA(n_components=8, svd_solver="randomized")
pca_LE_YY_l1 = PCA(n_components=8, svd_solver="randomized")

pca_LE_XX_s5 = PCA(n_components=8, svd_solver="randomized")
pca_LE_XX_m5 = PCA(n_components=8, svd_solver="randomized")
pca_LE_XX_l5 = PCA(n_components=8, svd_solver="randomized")

pca_LE_YY_s5 = PCA(n_components=8, svd_solver="randomized")
pca_LE_YY_m5 = PCA(n_components=8, svd_solver="randomized")
pca_LE_YY_l5 = PCA(n_components=8, svd_solver="randomized")

pca_LE_XX_s10 = PCA(n_components=8, svd_solver="randomized")
pca_LE_XX_m10 = PCA(n_components=8, svd_solver="randomized")
pca_LE_XX_l10 = PCA(n_components=8, svd_solver="randomized")

pca_LE_YY_s10 = PCA(n_components=8, svd_solver="randomized")
pca_LE_YY_m10 = PCA(n_components=8, svd_solver="randomized")
pca_LE_YY_l10 = PCA(n_components=8, svd_solver="randomized")

pca_LE_XX_s1.fit(LE_XX_scaled_s1)
pca_LE_XX_m1.fit(LE_XX_scaled_m1)
pca_LE_XX_l1.fit(LE_XX_scaled_l1)

pca_LE_YY_s1.fit(LE_YY_scaled_s1)
pca_LE_YY_m1.fit(LE_YY_scaled_m1)
pca_LE_YY_l1.fit(LE_YY_scaled_l1)

pca_LE_XX_s5.fit(LE_XX_scaled_s5)
pca_LE_XX_m5.fit(LE_XX_scaled_m5)
pca_LE_XX_l5.fit(LE_XX_scaled_l5)

pca_LE_YY_s5.fit(LE_YY_scaled_s5)
pca_LE_YY_m5.fit(LE_YY_scaled_m5)
pca_LE_YY_l5.fit(LE_YY_scaled_l5)

pca_LE_XX_s10.fit(LE_XX_scaled_s10)
pca_LE_XX_m10.fit(LE_XX_scaled_m10)
pca_LE_XX_l10.fit(LE_XX_scaled_l10)

pca_LE_YY_s10.fit(LE_YY_scaled_s10)
pca_LE_YY_m10.fit(LE_YY_scaled_m10)
pca_LE_YY_l10.fit(LE_YY_scaled_l10)

# Let's save components to file to train on another python file
LE_XX_comps_train_s1 = pca_LE_XX_s1.transform(LE_XX_scaled_s1)[:, 0:8]
LE_XX_comps_train_m1 = pca_LE_XX_m1.transform(LE_XX_scaled_m1)[:, 0:8]
LE_XX_comps_train_l1 = pca_LE_XX_l1.transform(LE_XX_scaled_l1)[:, 0:8]
LE_YY_comps_train_s1 = pca_LE_YY_s1.transform(LE_YY_scaled_s1)[:, 0:8]
LE_YY_comps_train_m1 = pca_LE_YY_m1.transform(LE_YY_scaled_m1)[:, 0:8]
LE_YY_comps_train_l1 = pca_LE_YY_l1.transform(LE_YY_scaled_l1)[:, 0:8]

LE_XX_comps_test_s1 = pca_LE_XX_s1.transform(LE_XX_test_scaled_s1)[:, 0:8]
LE_XX_comps_test_m1 = pca_LE_XX_m1.transform(LE_XX_test_scaled_m1)[:, 0:8]
LE_XX_comps_test_l1 = pca_LE_XX_l1.transform(LE_XX_test_scaled_l1)[:, 0:8]
LE_YY_comps_test_s1 = pca_LE_YY_s1.transform(LE_YY_test_scaled_s1)[:, 0:8]
LE_YY_comps_test_m1 = pca_LE_YY_m1.transform(LE_YY_test_scaled_m1)[:, 0:8]
LE_YY_comps_test_l1 = pca_LE_YY_l1.transform(LE_YY_test_scaled_l1)[:, 0:8]

LE_XX_comps_train_s5 = pca_LE_XX_s5.transform(LE_XX_scaled_s5)[:, 0:8]
LE_XX_comps_train_m5 = pca_LE_XX_m5.transform(LE_XX_scaled_m5)[:, 0:8]
LE_XX_comps_train_l5 = pca_LE_XX_l5.transform(LE_XX_scaled_l5)[:, 0:8]
LE_YY_comps_train_s5 = pca_LE_YY_s5.transform(LE_YY_scaled_s5)[:, 0:8]
LE_YY_comps_train_m5 = pca_LE_YY_m5.transform(LE_YY_scaled_m5)[:, 0:8]
LE_YY_comps_train_l5 = pca_LE_YY_l5.transform(LE_YY_scaled_l5)[:, 0:8]

LE_XX_comps_test_s5 = pca_LE_XX_s5.transform(LE_XX_test_scaled_s5)[:, 0:8]
LE_XX_comps_test_m5 = pca_LE_XX_m5.transform(LE_XX_test_scaled_m5)[:, 0:8]
LE_XX_comps_test_l5 = pca_LE_XX_l5.transform(LE_XX_test_scaled_l5)[:, 0:8]
LE_YY_comps_test_s5 = pca_LE_YY_s5.transform(LE_YY_test_scaled_s5)[:, 0:8]
LE_YY_comps_test_m5 = pca_LE_YY_m5.transform(LE_YY_test_scaled_m5)[:, 0:8]
LE_YY_comps_test_l5 = pca_LE_YY_l5.transform(LE_YY_test_scaled_l5)[:, 0:8]

LE_XX_comps_train_s10 = pca_LE_XX_s10.transform(LE_XX_scaled_s10)[:, 0:8]
LE_XX_comps_train_m10 = pca_LE_XX_m10.transform(LE_XX_scaled_m10)[:, 0:8]
LE_XX_comps_train_l10 = pca_LE_XX_l10.transform(LE_XX_scaled_l10)[:, 0:8]
LE_YY_comps_train_s10 = pca_LE_YY_s10.transform(LE_YY_scaled_s10)[:, 0:8]
LE_YY_comps_train_m10 = pca_LE_YY_m10.transform(LE_YY_scaled_m10)[:, 0:8]
LE_YY_comps_train_l10 = pca_LE_YY_l10.transform(LE_YY_scaled_l10)[:, 0:8]

LE_XX_comps_test_s10 = pca_LE_XX_s10.transform(LE_XX_test_scaled_s10)[:, 0:8]
LE_XX_comps_test_m10 = pca_LE_XX_m10.transform(LE_XX_test_scaled_m10)[:, 0:8]
LE_XX_comps_test_l10 = pca_LE_XX_l10.transform(LE_XX_test_scaled_l10)[:, 0:8]
LE_YY_comps_test_s10 = pca_LE_YY_s10.transform(LE_YY_test_scaled_s10)[:, 0:8]
LE_YY_comps_test_m10 = pca_LE_YY_m10.transform(LE_YY_test_scaled_m10)[:, 0:8]
LE_YY_comps_test_l10 = pca_LE_YY_l10.transform(LE_YY_test_scaled_l10)[:, 0:8]

# Save the data to file
param_train_df.to_csv("param_train.csv", header=False)
param_test_df.to_csv("param_test.csv", header=False)

pd.DataFrame(LE_XX_comps_train_s1, index=param_train_df.index).to_csv("LE_XX_comps_train_s1.csv", header=False)
pd.DataFrame(LE_XX_comps_train_m1, index=param_train_df.index).to_csv("LE_XX_comps_train_m1.csv", header=False)
pd.DataFrame(LE_XX_comps_train_l1, index=param_train_df.index).to_csv("LE_XX_comps_train_l1.csv", header=False)
pd.DataFrame(LE_YY_comps_train_s1, index=param_train_df.index).to_csv("LE_YY_comps_train_s1.csv", header=False)
pd.DataFrame(LE_YY_comps_train_m1, index=param_train_df.index).to_csv("LE_YY_comps_train_m1.csv", header=False)
pd.DataFrame(LE_YY_comps_train_l1, index=param_train_df.index).to_csv("LE_YY_comps_train_l1.csv", header=False)

pd.DataFrame(LE_XX_comps_test_s1, index=param_test_df.index).to_csv("LE_XX_comps_test_s1.csv", header=False)
pd.DataFrame(LE_XX_comps_test_m1, index=param_test_df.index).to_csv("LE_XX_comps_test_m1.csv", header=False)
pd.DataFrame(LE_XX_comps_test_l1, index=param_test_df.index).to_csv("LE_XX_comps_test_l1.csv", header=False)
pd.DataFrame(LE_YY_comps_test_s1, index=param_test_df.index).to_csv("LE_YY_comps_test_s1.csv", header=False)
pd.DataFrame(LE_YY_comps_test_m1, index=param_test_df.index).to_csv("LE_YY_comps_test_m1.csv", header=False)
pd.DataFrame(LE_YY_comps_test_l1, index=param_test_df.index).to_csv("LE_YY_comps_test_l1.csv", header=False)

pd.DataFrame(LE_XX_comps_train_s5, index=param_train_df.index).to_csv("LE_XX_comps_train_s5.csv", header=False)
pd.DataFrame(LE_XX_comps_train_m5, index=param_train_df.index).to_csv("LE_XX_comps_train_m5.csv", header=False)
pd.DataFrame(LE_XX_comps_train_l5, index=param_train_df.index).to_csv("LE_XX_comps_train_l5.csv", header=False)
pd.DataFrame(LE_YY_comps_train_s5, index=param_train_df.index).to_csv("LE_YY_comps_train_s5.csv", header=False)
pd.DataFrame(LE_YY_comps_train_m5, index=param_train_df.index).to_csv("LE_YY_comps_train_m5.csv", header=False)
pd.DataFrame(LE_YY_comps_train_l5, index=param_train_df.index).to_csv("LE_YY_comps_train_l5.csv", header=False)

pd.DataFrame(LE_XX_comps_test_s5, index=param_test_df.index).to_csv("LE_XX_comps_test_s5.csv", header=False)
pd.DataFrame(LE_XX_comps_test_m5, index=param_test_df.index).to_csv("LE_XX_comps_test_m5.csv", header=False)
pd.DataFrame(LE_XX_comps_test_l5, index=param_test_df.index).to_csv("LE_XX_comps_test_l5.csv", header=False)
pd.DataFrame(LE_YY_comps_test_s5, index=param_test_df.index).to_csv("LE_YY_comps_test_s5.csv", header=False)
pd.DataFrame(LE_YY_comps_test_m5, index=param_test_df.index).to_csv("LE_YY_comps_test_m5.csv", header=False)
pd.DataFrame(LE_YY_comps_test_l5, index=param_test_df.index).to_csv("LE_YY_comps_test_l5.csv", header=False)

pd.DataFrame(LE_XX_comps_train_s10, index=param_train_df.index).to_csv("LE_XX_comps_train_s10.csv", header=False)
pd.DataFrame(LE_XX_comps_train_m10, index=param_train_df.index).to_csv("LE_XX_comps_train_m10.csv", header=False)
pd.DataFrame(LE_XX_comps_train_l10, index=param_train_df.index).to_csv("LE_XX_comps_train_l10.csv", header=False)
pd.DataFrame(LE_YY_comps_train_s10, index=param_train_df.index).to_csv("LE_YY_comps_train_s10.csv", header=False)
pd.DataFrame(LE_YY_comps_train_m10, index=param_train_df.index).to_csv("LE_YY_comps_train_m10.csv", header=False)
pd.DataFrame(LE_YY_comps_train_l10, index=param_train_df.index).to_csv("LE_YY_comps_train_l10.csv", header=False)

pd.DataFrame(LE_XX_comps_test_s10, index=param_test_df.index).to_csv("LE_XX_comps_test_s10.csv", header=False)
pd.DataFrame(LE_XX_comps_test_m10, index=param_test_df.index).to_csv("LE_XX_comps_test_m10.csv", header=False)
pd.DataFrame(LE_XX_comps_test_l10, index=param_test_df.index).to_csv("LE_XX_comps_test_l10.csv", header=False)
pd.DataFrame(LE_YY_comps_test_s10, index=param_test_df.index).to_csv("LE_YY_comps_test_s10.csv", header=False)
pd.DataFrame(LE_YY_comps_test_m10, index=param_test_df.index).to_csv("LE_YY_comps_test_m10.csv", header=False)
pd.DataFrame(LE_YY_comps_test_l10, index=param_test_df.index).to_csv("LE_YY_comps_test_l10.csv", header=False)

# Added plotting code here to visualize PCA explained variance since redoing PCA in the plots notebook would be too much
plt.rcParams["font.family"] = "Helvetica"

x = range(0, 8)

color_palette = {
    "s1": "#2a4d69",  # dark muted blue
    "s5": "#88b04b",  # mid muted green
    "s10": "#f4a582",  # light red-peach

    "m1": "#2a4d69",
    "m5": "#88b04b",
    "m10": "#f4a582",

    "l1": "#2a4d69",
    "l5": "#88b04b",
    "l10": "#f4a582",
}

# Cumulative variance ratios for LE_XX at different time points
LE_XX_cvr_s1 = np.cumsum(pca_LE_XX_s1.explained_variance_ratio_)
LE_XX_cvr_m1 = np.cumsum(pca_LE_XX_m1.explained_variance_ratio_)
LE_XX_cvr_l1 = np.cumsum(pca_LE_XX_l1.explained_variance_ratio_)

LE_XX_cvr_s5 = np.cumsum(pca_LE_XX_s5.explained_variance_ratio_)
LE_XX_cvr_m5 = np.cumsum(pca_LE_XX_m5.explained_variance_ratio_)
LE_XX_cvr_l5 = np.cumsum(pca_LE_XX_l5.explained_variance_ratio_)

LE_XX_cvr_s10 = np.cumsum(pca_LE_XX_s10.explained_variance_ratio_)
LE_XX_cvr_m10 = np.cumsum(pca_LE_XX_m10.explained_variance_ratio_)
LE_XX_cvr_l10 = np.cumsum(pca_LE_XX_l10.explained_variance_ratio_)

# Cumulative variance ratios for LE_YY at different time points
LE_YY_cvr_s1 = np.cumsum(pca_LE_YY_s1.explained_variance_ratio_)
LE_YY_cvr_m1 = np.cumsum(pca_LE_YY_m1.explained_variance_ratio_)
LE_YY_cvr_l1 = np.cumsum(pca_LE_YY_l1.explained_variance_ratio_)

LE_YY_cvr_s5 = np.cumsum(pca_LE_YY_s5.explained_variance_ratio_)
LE_YY_cvr_m5 = np.cumsum(pca_LE_YY_m5.explained_variance_ratio_)
LE_YY_cvr_l5 = np.cumsum(pca_LE_YY_l5.explained_variance_ratio_)

LE_YY_cvr_s10 = np.cumsum(pca_LE_YY_s10.explained_variance_ratio_)
LE_YY_cvr_m10 = np.cumsum(pca_LE_YY_m10.explained_variance_ratio_)
LE_YY_cvr_l10 = np.cumsum(pca_LE_YY_l10.explained_variance_ratio_)

# New datasets (noiseless observations)
LE_XX_new = np.array([0.60053372, 0.82578346, 0.91385874, 0.94562439, 0.96518969, 0.97850729, 0.98761679, 0.99426144])
LE_YY_new = np.array([0.6104609, 0.83133299, 0.91660501, 0.94986924, 0.97026403, 0.98156913, 0.98923916, 0.99402813])

# Font size settings
title_fontsize = 28
label_fontsize = 20
legend_fontsize = 16
ticks_fontsize = 20

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Plotting for LE_XX
# Straight lines for s data
ax1.plot(np.array(x) + 1, LE_XX_cvr_s1[x] * 100, label="s1", color=color_palette["s1"], linestyle='-')
ax1.plot(np.array(x) + 1, LE_XX_cvr_s5[x] * 100, label="s5", color=color_palette["s5"], linestyle='-')
ax1.plot(np.array(x) + 1, LE_XX_cvr_s10[x] * 100, label="s10", color=color_palette["s10"], linestyle='-')

# Dashed lines for m data
ax1.plot(np.array(x) + 1, LE_XX_cvr_m1[x] * 100, label="m1", color=color_palette["m1"], linestyle='--')
ax1.plot(np.array(x) + 1, LE_XX_cvr_m5[x] * 100, label="m5", color=color_palette["m5"], linestyle='--')
ax1.plot(np.array(x) + 1, LE_XX_cvr_m10[x] * 100, label="m10", color=color_palette["m10"], linestyle='--')

# Dotted lines for l data
ax1.plot(np.array(x) + 1, LE_XX_cvr_l1[x] * 100, label="l1", color=color_palette["l1"], linestyle=':')
ax1.plot(np.array(x) + 1, LE_XX_cvr_l5[x] * 100, label="l5", color=color_palette["l5"], linestyle=':')
ax1.plot(np.array(x) + 1, LE_XX_cvr_l10[x] * 100, label="l10", color=color_palette["l10"], linestyle=':')

# Noiseless data
ax1.plot(np.array(x) + 1, LE_XX_new * 100, label="Noiseless", linestyle='--', color='black')
ax1.set_xlabel("Components", fontsize=label_fontsize)
ax1.set_ylabel("% Explained Variance", fontsize=label_fontsize)
ax1.set_title("$LE_{XX}$ Cumulative Explained Variance", fontsize=title_fontsize)
ax1.tick_params(axis='both', which='major', labelsize=ticks_fontsize)

# Plotting for LE_YY
# Straight lines for s data
ax2.plot(np.array(x) + 1, LE_YY_cvr_s1[x] * 100, label="s1", color=color_palette["s1"], linestyle='-')
ax2.plot(np.array(x) + 1, LE_YY_cvr_s5[x] * 100, label="s5", color=color_palette["s5"], linestyle='-')
ax2.plot(np.array(x) + 1, LE_YY_cvr_s10[x] * 100, label="s10", color=color_palette["s10"], linestyle='-')

# Dashed lines for m data
ax2.plot(np.array(x) + 1, LE_YY_cvr_m1[x] * 100, label="m1", color=color_palette["m1"], linestyle='--')
ax2.plot(np.array(x) + 1, LE_YY_cvr_m5[x] * 100, label="m5", color=color_palette["m5"], linestyle='--')
ax2.plot(np.array(x) + 1, LE_YY_cvr_m10[x] * 100, label="m10", color=color_palette["m10"], linestyle='--')

# Dotted lines for l data
ax2.plot(np.array(x) + 1, LE_YY_cvr_l1[x] * 100, label="l1", color=color_palette["l1"], linestyle=':')
ax2.plot(np.array(x) + 1, LE_YY_cvr_l5[x] * 100, label="l5", color=color_palette["l5"], linestyle=':')
ax2.plot(np.array(x) + 1, LE_YY_cvr_l10[x] * 100, label="l10", color=color_palette["l10"], linestyle=':')

# Noiseless data
ax2.plot(np.array(x) + 1, LE_YY_new * 100, label="Noiseless", linestyle='--', color='black')
ax2.set_xlabel("Components", fontsize=label_fontsize)
ax2.set_ylabel("% Explained Variance", fontsize=label_fontsize)
ax2.set_title("$LE_{YY}$ Cumulative Explained Variance", fontsize=title_fontsize)
ax2.tick_params(axis='both', which='major', labelsize=ticks_fontsize)

# Create a single legend for both plots
plt.plot(np.array(x) + 1, LE_YY_new * 100, label="Noiseless", linestyle='--', color='black')
plt.xlabel("Components", fontsize=label_fontsize)
plt.ylabel("% Explained Variance", fontsize=label_fontsize)
plt.title("$LE_{YY}$ Cumulative Explained Variance", fontsize=title_fontsize)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.legend(bbox_to_anchor=(0.5, -0.1), loc='lower center', ncol=6, fontsize=legend_fontsize, bbox_transform=fig.transFigure)  # Legend below both plots

plt.show()

