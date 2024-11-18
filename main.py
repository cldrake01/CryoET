import zarr
import numpy as np

# Create a Zarr array in memory
z = zarr.array(np.random.rand(100, 100), chunks=(10, 10))

# Store the array on disk
zarr.save("my_array.zarr", z)

z = zarr.open("my_array.zarr", mode="r")
print(z.shape)
print(z[:10, :10])
