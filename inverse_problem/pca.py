import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA

# Read in strain components and parameters
parameter_df = pd.read_csv("Parameters.csv", header=None)
LE_XX = pd.read_csv("LE_XX.csv", header=None)
LE_YY = pd.read_csv("LE_YY.csv", header=None)

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

LE_XX_train_df = LE_XX.T.loc[train_indicies]
LE_XX_test_df = LE_XX.T.loc[test_indicies]

LE_YY_train_df = LE_YY.T.loc[train_indicies]
LE_YY_test_df = LE_YY.T.loc[test_indicies]

LE_XX_scalar = preprocessing.StandardScaler()
LE_YY_scalar = preprocessing.StandardScaler()

LE_XX_scaled = LE_XX_scalar.fit_transform(LE_XX_train_df)
LE_YY_scaled = LE_YY_scalar.fit_transform(LE_YY_train_df)
LE_XX_test_scaled = LE_XX_scalar.transform(LE_XX_test_df)
LE_YY_test_scaled = LE_YY_scalar.transform(LE_YY_test_df)

pca_LE_XX = PCA(n_components=8, svd_solver="randomized") # n_components=8
pca_LE_YY = PCA(n_components=8, svd_solver="randomized") # n_components=8
pca_LE_XX.fit(LE_XX_scaled)
pca_LE_YY.fit(LE_YY_scaled)

LE_XX_comps_train = pca_LE_XX.transform(LE_XX_scaled)
LE_YY_comps_train = pca_LE_YY.transform(LE_YY_scaled)
LE_XX_comps_test = pca_LE_XX.transform(LE_XX_test_scaled)
LE_YY_comps_test = pca_LE_YY.transform(LE_YY_test_scaled)

# Save the data to file
param_train_df.to_csv("param_train.csv", header=False)
param_test_df.to_csv("param_test.csv", header=False)
pd.DataFrame(LE_XX_comps_train, index=param_train_df.index).to_csv("LE_XX_comps_train.csv", header=False)
pd.DataFrame(LE_XX_comps_test, index=param_test_df.index).to_csv("LE_XX_comps_test.csv", header=False)
pd.DataFrame(LE_YY_comps_train, index=param_train_df.index).to_csv("LE_YY_comps_train.csv", header=False)
pd.DataFrame(LE_YY_comps_test, index=param_test_df.index).to_csv("LE_YY_comps_test.csv", header=False)
