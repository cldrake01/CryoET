import zarr
import numpy as np

# Store the array on disk
path = "CryoET Object Identification/test/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000"
z = zarr.open(path, mode="r")
print(z.shape)
print(z[:10, :10])
