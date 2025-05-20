# Load in out surrogate model
import pickle
import numpy as np
import torch
import pyro
import pandas as pd
import sys
import os
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# Check if correct number of arguments are provided
if len(sys.argv) != 4:
    print("Usage: python invsere_multi_expansion.py <noise_name> <data_use> <samples_to_view>")
    print("Example: python invsere_multi_expansion.py s1 0 13")
    sys.exit(1)

# Get arguments
noise_name = sys.argv[1]
data_use = int(sys.argv[2])
samples_to_view = int(sys.argv[3])

# Validate noise type
valid_types = ['s1', 's5', 's10', 'm1', 'm5', 'm10', 'l1', 'l5', 'l10']
if noise_name not in valid_types:
    print(f"Invalid noise type. Must be one of {valid_types}")
    sys.exit(1)

# Set up file paths
data_dir = f"{noise_name}_partitioned_data"
xx_gp_path = f"XX_GP_{noise_name}.pkl"
yy_gp_path = f"YY_GP_{noise_name}.pkl"

# Import data
labels = pd.read_csv(os.path.join(data_dir, f"{noise_name}_expansion_labels_{data_use}.csv"), header=None, index_col=0)
xx_targets = pd.read_csv(os.path.join(data_dir, f"{noise_name}_expansion_xx_{data_use}.csv"), header=None, index_col=0)
yy_targets = pd.read_csv(os.path.join(data_dir, f"{noise_name}_expansion_yy_{data_use}.csv"), header=None, index_col=0)

# Convert pandas DataFrames to torch Tensors
labels_tensor = torch.tensor(labels.values)
xx_targets_tensor = torch.tensor(xx_targets.values)
yy_targets_tensor = torch.tensor(yy_targets.values)

# Load GP models
with open(xx_gp_path, 'rb') as f:
    xx_gp_model = pickle.load(f)

with open(yy_gp_path, 'rb') as f:
    yy_gp_model = pickle.load(f)

# Update the index to use the command line argument
ind = samples_to_view

def multiComponentModel(alpha_1=None, alpha_2=None, xx_observations=None, yy_observations=None):

    # Define Prior
    mu = pyro.sample("mu", dist.Uniform(low=0.000543, high=0.00206))
    k1 = pyro.sample("k1", dist.Uniform(low=0.00117, high=0.0225)) # low=0.00117, high=0.0225
    k2 = pyro.sample("k2", dist.Uniform(low=8.31, high=15.32)) # low=8.31, high=15.32
    kappa = pyro.sample("kappa", dist.Uniform(low=0.27, high=0.3)) # low=0.27, high=0.3
    theta = pyro.sample("theta", dist.Uniform(low=0, high=90)) # low=0, high=90)
    
    # Alphas are deterministic
    a1 = alpha_1
    a2 = alpha_2

    xx_var1 = pyro.sample("xx_var1", dist.HalfCauchy(scale=2.0))
    xx_var2 = pyro.sample("xx_var2", dist.HalfCauchy(scale=2.0))
    xx_var3 = pyro.sample("xx_var3", dist.HalfCauchy(scale=2.0))
    xx_var4 = pyro.sample("xx_var4", dist.HalfCauchy(scale=1.0))
    xx_var5 = pyro.sample("xx_var5", dist.HalfCauchy(scale=1.0))
    xx_var6 = pyro.sample("xx_var6", dist.HalfCauchy(scale=0.5))
    xx_var7 = pyro.sample("xx_var7", dist.HalfCauchy(scale=0.5))
    xx_var8 = pyro.sample("xx_var8", dist.HalfCauchy(scale=0.5))

    yy_var1 = pyro.sample("yy_var1", dist.HalfCauchy(scale=2.0))
    yy_var2 = pyro.sample("yy_var2", dist.HalfCauchy(scale=2.0))
    yy_var3 = pyro.sample("yy_var3", dist.HalfCauchy(scale=2.0))
    yy_var4 = pyro.sample("yy_var4", dist.HalfCauchy(scale=1.0))
    yy_var5 = pyro.sample("yy_var5", dist.HalfCauchy(scale=1.0))
    yy_var6 = pyro.sample("yy_var6", dist.HalfCauchy(scale=1.0))
    yy_var7 = pyro.sample("yy_var7", dist.HalfCauchy(scale=0.5))
    yy_var8 = pyro.sample("yy_var8", dist.HalfCauchy(scale=0.5))

    for i, [a1, a2] in enumerate(zip(alpha_1, alpha_2)):
        input_var = torch.tensor([mu, k1, k2, kappa, theta, a1, a2]).reshape(1,-1)

        # SKlearn GP
        xx_predictions = xx_gp_model.predict(input_var.reshape(1,-1))[0].reshape(1,-1)[0]
        yy_predictions = yy_gp_model.predict(input_var.reshape(1,-1))[0].reshape(1,-1)[0]
        
        # Define likelihood
        pyro.sample(f"zx1_{i}", dist.Normal(xx_predictions[0], xx_var1), obs=xx_observations[i][0])
        pyro.sample(f"zx2_{i}", dist.Normal(xx_predictions[1], xx_var2), obs=xx_observations[i][1])
        pyro.sample(f"zx3_{i}", dist.Normal(xx_predictions[2], xx_var3), obs=xx_observations[i][2])
        pyro.sample(f"zx4_{i}", dist.Normal(xx_predictions[3], xx_var4), obs=xx_observations[i][3])
        pyro.sample(f"zx5_{i}", dist.Normal(xx_predictions[4], xx_var5), obs=xx_observations[i][4])
        pyro.sample(f"zx6_{i}", dist.Normal(xx_predictions[5], xx_var6), obs=xx_observations[i][5])
        pyro.sample(f"zx7_{i}", dist.Normal(xx_predictions[6], xx_var7), obs=xx_observations[i][6])
        pyro.sample(f"zx8_{i}", dist.Normal(xx_predictions[7], xx_var8), obs=xx_observations[i][7])

        pyro.sample(f"zy1_{i}", dist.Normal(yy_predictions[0], yy_var1), obs=yy_observations[i][0])
        pyro.sample(f"zy2_{i}", dist.Normal(yy_predictions[1], yy_var2), obs=yy_observations[i][1])
        pyro.sample(f"zy3_{i}", dist.Normal(yy_predictions[2], yy_var3), obs=yy_observations[i][2])
        pyro.sample(f"zy4_{i}", dist.Normal(yy_predictions[3], yy_var4), obs=yy_observations[i][3])
        pyro.sample(f"zy5_{i}", dist.Normal(yy_predictions[4], yy_var5), obs=yy_observations[i][4])
        pyro.sample(f"zy6_{i}", dist.Normal(yy_predictions[5], yy_var6), obs=yy_observations[i][5])
        pyro.sample(f"zy7_{i}", dist.Normal(yy_predictions[6], yy_var7), obs=yy_observations[i][6])
        pyro.sample(f"zy8_{i}", dist.Normal(yy_predictions[7], yy_var8), obs=yy_observations[i][7])


# MCMC for the component model
nuts_kernel = NUTS(multiComponentModel, target_accept_prob=0.1)
mcmc = MCMC(nuts_kernel, warmup_steps=10000, num_samples=10000)

a11 = labels_tensor[-samples_to_view:, 5]
a22 = labels_tensor[-samples_to_view:, 6]
xx_obs = xx_targets_tensor[-samples_to_view:]
yy_obs = yy_targets_tensor[-samples_to_view:]

mcmc.run(a11, a22, xx_obs, yy_obs)
mcmc.summary()

comp_samples = mcmc.get_samples(10000)

# Save our samples to a csv file
samples_pd = pd.DataFrame(comp_samples)
samples_pd.to_csv(f"MCMC{noise_name}_Multi_Expansion{data_use}_{samples_to_view}.csv")