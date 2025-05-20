import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Specify noise level for different scales
noise_level = 10  # 1, 5, or 10 for our paper's results

# Read in a mesh from Abaqus
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

# string_to_list(LE_strain[0,0])[0]
node_X = np.array(node_X_array)
elements = np.array(elem_array)

# Only first 3748 elements are the top surface, so get those top nodes which form those elements
surf_elements = elements[0:3748][:, 0:4]
coords = node_X[surf_elements - 1] # -1 is for indexing

# Get coordinates of centroid of each element
x_coords = np.array([])
y_coords = np.array([])

for single_element in coords:
    x_avg = np.mean(single_element[:, 0])
    y_avg = np.mean(single_element[:, 2])
    x_coords = np.append(x_coords, x_avg)
    y_coords = np.append(y_coords, y_avg)

# Define coordinates (centers of each element)
x = x_coords
y = y_coords
X = np.column_stack((x, y))  # Create an array of (x, y) pairs

# Define a kernel with a specific length scale for correlation between closer points
kernel_small = C(1.0) * RBF(length_scale=0.1)
kernel_medium = C(1.0) * RBF(length_scale=1)
kernel_large = C(1.0) * RBF(length_scale=5)

# Create a Gaussian Process model (without fitting)
gp_small = GaussianProcessRegressor(kernel=kernel_small)
gp_medium = GaussianProcessRegressor(kernel=kernel_medium)
gp_large = GaussianProcessRegressor(kernel=kernel_large)

# Directly sample from the GP's prior distribution
noise_small_xx = gp_small.sample_y(X, n_samples=16800, random_state=422)
noise_small_yy = gp_small.sample_y(X, n_samples=16800, random_state=375)
noise_medium_xx = gp_medium.sample_y(X, n_samples=16800, random_state=422)
noise_medium_yy = gp_medium.sample_y(X, n_samples=16800, random_state=375)
noise_large_xx = gp_large.sample_y(X, n_samples=16800, random_state=422)
noise_large_yy = gp_large.sample_y(X, n_samples=16800, random_state=375)

# Load data
LE_XX = pd.read_csv("LE_XX.csv", header=None)
LE_YY = pd.read_csv("LE_YY.csv", header=None)

# Convert to numpy array
LE_XX_np = np.array(LE_XX.values)
LE_YY_np = np.array(LE_YY.values)

# Find maximum value of strain
x_max = LE_XX_np.max()
y_max = LE_YY_np.max()

# Scale the noise by a specified scale factor
def scale_noise(data, scale):
    return data * scale

scaler = noise_level * 0.01
magnitude_scale_xx = x_max * scaler # 10% of max value
magnitude_scale_yy = y_max * scaler # 10% of max value

small_scaled_noise_xx = scale_noise(noise_small_xx, magnitude_scale_xx) 
small_scaled_noise_yy = scale_noise(noise_small_yy, magnitude_scale_yy) 
medium_scaled_noise_xx = scale_noise(noise_medium_xx, magnitude_scale_xx)
medium_scaled_noise_yy = scale_noise(noise_medium_yy, magnitude_scale_yy)
large_scaled_noise_xx = scale_noise(noise_large_xx, magnitude_scale_xx)
large_scaled_noise_yy = scale_noise(noise_large_yy, magnitude_scale_yy)

# Add in noise to data
small_noisy_LE_xx = LE_XX_np + small_scaled_noise_xx
medium_noisy_LE_xx = LE_XX_np + medium_scaled_noise_xx
large_noisy_LE_xx = LE_XX_np + large_scaled_noise_xx

small_noisy_LE_yy = LE_YY_np + small_scaled_noise_yy
medium_noisy_LE_yy = LE_YY_np + medium_scaled_noise_yy
large_noisy_LE_yy = LE_YY_np + large_scaled_noise_yy

# Save our noisy data for analysis
pd.DataFrame(small_noisy_LE_xx).to_csv(f"LE_XX_{noise_level}s.csv", header=False, index=False)
pd.DataFrame(medium_noisy_LE_xx).to_csv(f"LE_XX_{noise_level}m.csv", header=False, index=False)
pd.DataFrame(large_noisy_LE_xx).to_csv(f"LE_XX_{noise_level}l.csv", header=False, index=False)

pd.DataFrame(small_noisy_LE_yy).to_csv(f"LE_YY_{noise_level}s.csv", header=False, index=False)
pd.DataFrame(medium_noisy_LE_yy).to_csv(f"LE_YY_{noise_level}m.csv", header=False, index=False)
pd.DataFrame(large_noisy_LE_yy).to_csv(f"LE_YY_{noise_level}l.csv", header=False, index=False)