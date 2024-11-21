import os
import numpy as np
from typing import List, Dict, Any, Optional

from abstract.abstract import AbstractMultimodalDataset

class ExampleMultimodalDataset(AbstractMultimodalDataset):
    def __init__(self, dataset_name: str, modalities: List[str], data_dir: str):
        super().__init__(dataset_name, modalities, data_dir)
        self.data = {}  # Store data, e.g., {'image': [...], 'text': [...]}

    def load_data(self) -> None:
        # Simulate loading image and text data
        self.data['image'] = [np.random.rand(256, 256, 3) for _ in range(100)]  # Simulated image data
        self.data['text'] = [f"Sample text {i}" for i in range(100)]  # Simulated text data

    def get_sample(self, index: int) -> Dict[str, Any]:
        return {modality: self.data[modality][index] for modality in self.modalities}

    def __len__(self) -> int:
        # Assume all modalities have the same number of samples
        return len(self.data['image'])

# Example usage
dataset = ExampleMultimodalDataset("ExampleDataset", ["image", "text"], "/path/to/data")
dataset.load_data()
print(dataset.get_sample(0))  # Retrieve the first sample
print(dataset.get_metadata())  # Print metadata
