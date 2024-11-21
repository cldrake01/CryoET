import os
from typing import Iterable

import torch
import zarr
from torch import Tensor
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
    def f(path: str) -> Group:
        group = zarr.open(path, mode="r")
        # Uncomment to print the group info and tree
        # print("\n".join((f"{group.info = }", f"{group.tree() = }")))
        return group

    return [
        f(os.path.join(root, folder))
        for root, folders, _ in os.walk(directory)
        for folder in folders
        if folder.endswith(".zarr")
    ]


def tensors_from(groups: Iterable[Group]) -> list[Tensor]:
    # Old
    # return [torch.tensor(group[x][:]) for x in group.array_keys()]
    return [
        torch.tensor(group.array_keys()[index][:]) for index, group in enumerate(groups)
    ]


groups = groups_from(directory)
tensors = tensors_from(groups[0])

dataset = ZarrDataset(
    dict(
        modality="images",
        filenames=groups,
        source_axes="ZYX",
        data_group="0",
    )
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for x in dataloader:
    print(x.shape)
    break
