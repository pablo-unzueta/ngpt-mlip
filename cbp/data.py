from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import Union, Tuple
import torch


def return_train_val_test_loader(
    path: Union[str, Path],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # check if path is a directory
    if not isinstance(path, Path):
        path = Path(path)

    # check if the files exist
    if not (path / "train_dataset.pt").exists():
        raise FileNotFoundError(f"File {path / 'train_dataset.pt'} does not exist")
    if not (path / "val_dataset.pt").exists():
        raise FileNotFoundError(f"File {path / 'val_dataset.pt'} does not exist")
    if not (path / "test_dataset.pt").exists():
        raise FileNotFoundError(f"File {path / 'test_dataset.pt'} does not exist")

    train_dataset = torch.load(path / "train_dataset.pt", weights_only=False)
    val_dataset = torch.load(path / "val_dataset.pt", weights_only=False)
    test_dataset = torch.load(path / "test_dataset.pt", weights_only=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, val_loader, test_loader
