import torch
from typing import Callable
from torch.nn import Module
from torch.utils.data import Dataset

class BaseTrainer:
    def __init__(self, model: Module, test_data: Dataset, transforms: Callable, n_views: int = 3):
        """Initialise with SSL model, test data, augmentation transforms, and view count."""
        self.ssl_model = model
        self.test_data = test_data
        self.transforms = transforms
        self.n_views = n_views
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssl_model = self.ssl_model.to(self.device)

    def test_loop(self, base_model: Module, adapt_iters: int = 5):
        """Override in subclasses to run test-time adaptation."""
        raise NotImplementedError("Subclasses must implement this method!")