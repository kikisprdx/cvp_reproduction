import torch
from typing import Callable
from torch.nn import Module
from torch.utils.data import Dataset

class BaseTrainer:
    def __init__(self, model: Module, test_data: Dataset, transforms: Callable, n_views: int = 3):
        """
        the foundation for all test-time adaptation trainers.
        
        Args:
            model: The pre-trained SSL MLP model.
            test_data: DataLoader for the corrupted test dataset.
            transforms: the transformations for contrastive learning; ( usually shared SSL transformations so crop, flip, rotate)
            n_views: number of augmented views to generate.
        """
        self.ssl_model = model
        self.test_data = test_data
        self.transforms = transforms
        self.n_views = n_views
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssl_model = self.ssl_model.to(self.device)

    def test_loop(self, base_model: Module, adapt_iters: int = 5):
        """The core adaptation and inference loop (?)"""
        raise NotImplementedError("Subclasses must implement this method!")