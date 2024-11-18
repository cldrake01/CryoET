import zarr
import numpy as np

zarr_file = zarr.open('/Users/jackcerullo/Documents/GitHub/CryoET/CryoET Object Identification/train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr', mode='r')
zarr_array = zarr_file[0][0][0][0]
numpy_array = zarr_array[:]
print(numpy_array)
