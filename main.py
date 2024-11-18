import zarr
import numpy as np

# Create a Zarr array in memory
z = zarr.array(np.random.rand(100, 100), chunks=(10, 10))

# Store the array on disk
path = "CryoET Object Identification/test/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000"
zarr.save(path, z)

z = zarr.open(path, mode="r")
print(z.shape)
print(z[:10, :10])
