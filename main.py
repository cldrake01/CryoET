import logging
import os

import zarr
from torch.utils.data import DataLoader
from zarr import Group
from zarrdataset import ZarrDataset

# Define the relative path to the base directory
directory = "CryoET Object Identification/train/static/"


# def zarr_directories(directory: str) -> list[str]:
#     return [
#         os.path.join(root, folder)
#         for root, folders, _ in os.walk(directory)
#         for folder in folders
#         if folder.endswith(".zarr")
#     ]


def groups_from(directory: str) -> list[Group]:
    # zarr_dirs = []
    # for root, folders, _ in os.walk(directory):
    #     for folder in folders:
    #         if folder.endswith(".zarr"):
    #             g = zarr.open_group(os.path.join(root, folder))
    #             print(f"Opened {os.path.join(root, folder)}")
    #             print(f"{g.tree() = },\n{g.info = }")
    #             zarr_dirs.append(g)
    # return zarr_dirs
    # Use a list comprehension to open all zarr files in the directory
    def f(x):
        print(f"Opened {x}")
        logging.info(f"Opened {x}")
        return zarr.open_group(x)

    return [
        f(os.path.join(root, folder))
        for root, folders, _ in os.walk(directory)
        for folder in folders
        if folder.endswith(".zarr")
    ]


groups = groups_from(directory)

dataset = ZarrDataset(
    dict(
        modality="images",
        filenames=groups,
        source_axes="ZYX",
        data_group="0",
    )
)

dataloader = DataLoader(dataset)

for x in dataloader:
    print(x.shape)
    break
