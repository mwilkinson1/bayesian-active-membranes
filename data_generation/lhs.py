from scipy.stats import qmc
import matplotlib.pyplot as plt
import csv
import numpy as np

# Initialize param set
params = []

# Read in data (Skin parameters from )
with open("skin_parameters.csv", 'r') as param_file:
    reader = csv.reader(param_file, delimiter=',') 
    for row in reader:
        params.append(row) # Read each row and add it to params
        
# Get only the in vivo conditions and format so that each row is the range
in_vivo = np.array(params[5:9]).T

# Get the max and min of each param
max_params = []
min_params = []
for element in in_vivo:
    max_params.append(max(map(float,element))) # Get the max element from each set
    min_params.append(min(map(float,element))) # Get the min element from each set

# Add the angle variation into the max and min sets
max_params.append(90)
min_params.append(0)

# Expansion terms alpha_11 and alpha_22
alpha_params = []
alpha_pairs = set()
alpha_scale = np.linspace(-0.05, 0.05, 7)

for element in alpha_scale:
    rounded_element = element.round(9)
    alpha_pairs.add((0.05, rounded_element))
    alpha_pairs.add((-0.05, rounded_element))

for element in alpha_pairs:
    alpha_params.append(list(element))

x_points = []
y_points = []
for element in alpha_pairs:
    x_points.append(element[0])
    y_points.append(element[1])

# Latin Hypercube Sampling
sampler = qmc.LatinHypercube(d=5, seed=122)

n_samples = 120 # Number of samples that we need
sample = sampler.random(n_samples)

# Scale up to get the true sample parameters
sample_scaled = qmc.scale(sample, min_params, max_params)

# Check that angles are correct
angles = np.zeros(len(sample_scaled))
for ind, element in enumerate(sample_scaled):
    angles[ind] = element[-1] * (np.pi / 180)

# Write to save to csv
with open("Expanded In Vivo Parameters.csv", 'w') as file:
    csv_writer = csv.writer(file)
    for sample_set in sample_scaled:
        csv_writer.writerow(sample_set)