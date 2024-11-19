# import os

# directory_path = '/Users/jackcerullo/Documents/GitHub/CryoET/CryoET Object Identification/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000'
# print()
# for root, dirs, files in os.walk(directory_path):
#     print(f"Root: {root}")
#     print(f"Dirs: {dirs}")
#     print(f"Files: {files}")
import os
import zarr
import torch
import numpy as np

# Define the path to the main folder
base_dir = "/Users/jackcerullo/Documents/GitHub/CryoET/CryoET Object Identification/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000"

# Example of a function to load a .zarr file into a PyTorch tensor
def load_zarr_to_tensor(zarr_path):
    # Open the .zarr file using zarr
    store = zarr.open(zarr_path, mode='r')

    # Find the .zarray file, which contains the data
    # This assumes that the data is stored in the root of the .zarr file
    data = store[:]

    # Convert to numpy array, then to a PyTorch tensor
    return torch.tensor(data)

# Example: Loading data from 'isonetcorrected.zarr'
zarr_file_path = os.path.join(base_dir, "isonetcorrected.zarr")
tensor_data = load_zarr_to_tensor(zarr_file_path)

# Print the shape of the tensor
print(tensor_data.shape)
