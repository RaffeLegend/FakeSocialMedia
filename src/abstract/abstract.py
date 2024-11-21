from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class AbstractMultimodalDataset(ABC):
    """
    Abstract class for managing multimodal datasets, including basic dataset properties and methods.
    Designed to be extensible for further development.
    """
    def __init__(self, dataset_name: str, modalities: List[str], data_dir: str):
        """
        Initialize basic attributes of the multimodal dataset.

        Args:
            dataset_name (str): The name of the dataset.
            modalities (List[str]): List of modalities (e.g., ['image', 'text', 'audio']).
            data_dir (str): Root directory where the dataset is stored.
        """
        self.dataset_name = dataset_name
        self.modalities = modalities
        self.data_dir = data_dir
        self.metadata: Optional[Dict[str, Any]] = None  # Metadata for the dataset

    @abstractmethod
    def load_data(self) -> None:
        """
        Abstract method: Load the dataset.
        Subclasses should implement specific data loading logic.
        """
        pass

    @abstractmethod
    def get_sample(self, index: int) -> Dict[str, Any]:
        """
        Abstract method: Retrieve a single sample by index.

        Args:
            index (int): Index of the sample.

        Returns:
            Dict[str, Any]: Dictionary containing data from multiple modalities, 
                            e.g., {'image': image_data, 'text': text_data}.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Abstract method: Return the size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve metadata of the dataset.

        Returns:
            Dict[str, Any]: Metadata such as {'num_samples': 1000, 'modalities': ['image', 'text']}.
        """
        if self.metadata is None:
            self.metadata = {
                "dataset_name": self.dataset_name,
                "num_samples": len(self),
                "modalities": self.modalities,
                "data_dir": self.data_dir
            }
        return self.metadata

    def filter_data(self, condition: Any) -> List[int]:
        """
        Interface: Filter samples based on specific conditions (extensible).

        Args:
            condition (Any): Filtering condition.

        Returns:
            List[int]: List of indices that meet the condition.
        """
        raise NotImplementedError("This method should be implemented in a subclass if needed.")

    def transform(self, modality: str, transform_func: Any) -> None:
        """
        Interface: Apply transformations to a specific modality (extensible).

        Args:
            modality (str): Name of the modality, e.g., 'image' or 'text'.
            transform_func (Any): Transformation function to be applied.
        """
        raise NotImplementedError("This method should be implemented in a subclass if needed.")

    def split_data(self, split_ratio: List[float]) -> List[Any]:
        """
        Interface: Split the dataset into training, validation, and test sets (extensible).

        Args:
            split_ratio (List[float]): Ratios for splitting, e.g., [0.8, 0.1, 0.1].

        Returns:
            List[Any]: List of datasets after splitting.
        """
        raise NotImplementedError("This method should be implemented in a subclass if needed.")
