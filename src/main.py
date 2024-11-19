import zarr
import numpy as np

# Store the array on disk
path = "CryoET Object Identification/test/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr"
print(path)
z = zarr.group(store=path, overwrite=False).create_group('denoised').create_dataset('data', shape=(100, 100, 100), chunks=(10, 10, 10), dtype='f4')
print(z.shape)
print(z[:10, :10])
