import os
from torch.utils.data import DataLoader
from lightly.data import LightlyDataset

def get_pretrain_dataloader(data_dir: str, transform, batch_size: int, num_workers: int = 4):
    """
    Wraps the dataset using LightlyDataset to handle multiple views and returns the DataLoader.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    # LightlyDataset automatically handles the nested folder structure
    dataset = LightlyDataset(input_dir=data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    
    return dataloader