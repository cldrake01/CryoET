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
base_dir = "/Users/jackcerullo/Documents/GitHub/CryoET/CryoET Object Identification/train/static/ExperimentRuns/"
copy_path = "/Users/jackcerullo/Documents/GitHub/CryoET/CryoET Object Identification copy/train/static/ExperimentRuns"


# Example of a function to load a .zarr file into a PyTorch tensor
def load_zarr_to_tensor(zarr_path):
    # Open the .zarr file using zarr
    store = zarr.open_group(copy_path, mode="w")
    store.create_dataset("train", shape=(512, 512), data=0)
    print(store["train"])
    print(*store.items())
    print("that worked")

    # Find the .zarray file, which contains the data
    # This assumes that the data is stored in the root of the .zarr file

    # Convert to numpy array, then to a PyTorch tensor
    # return torch.tensor(data)


# Example: Loading data from 'isonetcorrected.zarr'
zarr_file_path = os.path.join(base_dir, "isonetcorrected.zarr")
tensor_data = load_zarr_to_tensor(zarr_file_path)

# Print the shape of the tensor
print(tensor_data.shape)
