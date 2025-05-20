import itertools
import os

# Define all possible input values
noise_types = ['s1', 's5', 's10', 'm1', 'm5', 'm10', 'l1', 'l5', 'l10']
expansion_types = ['max', 'mixed', 'uni']
time_samples_to_view = list(range(1, 11))
expansion_samples_to_view = list(range(1, 15))
param_nums = [0, 1, 2]  # This will be used as the index parameter for single expansion files
single_obs = list(range(0, 20))

# Create task_files directory if it doesn't exist
os.makedirs('task_files', exist_ok=True)

# Generate multi-time tasks
multi_time_tasks = []
for noise, exp_type, samples, param in itertools.product(
    noise_types, expansion_types, time_samples_to_view, param_nums
):
    multi_time_tasks.append(f"python inverse_multi_time.py {noise} {exp_type} {samples} {param}")

# Generate multi-expansion tasks
multi_expansion_tasks = []
for noise, exp_type, samples, param in itertools.product(
    noise_types, expansion_types, expansion_samples_to_view, param_nums
):
    multi_expansion_tasks.append(f"python inverse_multi_expansion.py {noise} {exp_type} {samples} {param}")

# Generate single expansion tasks for each type
single_max_tasks = []
single_mixed_tasks = []
single_uniaxial_tasks = []

for noise, obs_num in itertools.product(noise_types, single_obs):
    single_max_tasks.append(f"python inverse_single_max.py {noise} {obs_num}")
    single_mixed_tasks.append(f"python inverse_single_mixed.py {noise} {obs_num}")
    single_uniaxial_tasks.append(f"python inverse_single_uniaxial.py {noise} {obs_num}")

# Write all tasks to their respective text files
with open('task_files/inverse_multi_time_tasks.txt', 'w') as f:
    f.write("\n".join(multi_time_tasks))

with open('task_files/inverse_multi_expansion_tasks.txt', 'w') as f:
    f.write("\n".join(multi_expansion_tasks))

with open('task_files/inverse_single_max_tasks.txt', 'w') as f:
    f.write("\n".join(single_max_tasks))

with open('task_files/inverse_single_mixed_tasks.txt', 'w') as f:
    f.write("\n".join(single_mixed_tasks))

with open('task_files/inverse_single_uniaxial_tasks.txt', 'w') as f:
    f.write("\n".join(single_uniaxial_tasks))

# Print summary
print(f"Generated task files in 'task_files' directory:")
print(f"- inverse_multi_time_tasks.txt: {len(multi_time_tasks)} tasks")
print(f"- inverse_multi_expansion_tasks.txt: {len(multi_expansion_tasks)} tasks")
print(f"- inverse_single_max_tasks.txt: {len(single_max_tasks)} tasks")
print(f"- inverse_single_mixed_tasks.txt: {len(single_mixed_tasks)} tasks")
print(f"- inverse_single_uniaxial_tasks.txt: {len(single_uniaxial_tasks)} tasks")
