import copy
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from models.CVP_model import CVP
from utils import contrastive_loss


class CVPTrainer:
    def __init__(
        self,
        model: CVP,
        optimiser,
        training_data,
        test_data,
        tau=0.2,
        transforms=None,
        n_views=2,
    ):
        self.model = model
        self.optimiser = optimiser
        self.tau = tau

        self.training_data = training_data
        self.test_data = test_data

        self.transforms = transforms
        self.n_views = n_views

        self.base_path = "models/cvp/"
        os.makedirs(self.base_path, exist_ok=True)
        self.best_model_path = os.path.join(self.base_path, "cvp")

    def online_train(self, data, num_iterations, transforms, n_views, device):
        temp_model = copy.deepcopy(self.model)
        temp_optimiser = torch.optim.Adam(temp_model.head.parameters(), lr=1e-4)
        temp_model.train()
        # For every single x, it creates 2
        # possibly augmented version of x
        # does this for every batch
        # -> model evaluates on this
        with torch.no_grad():
            views = torch.cat([transforms(data.cpu()) for _ in range(n_views)], dim=0).to(device)
            initial_loss = contrastive_loss(temp_model(views), data.size(0), n_views=n_views, temperature=self.tau)

        for _ in range(num_iterations):
            views = torch.cat([transforms(data.cpu()) for _ in range(n_views)], dim=0).to(device)
            new_loss = contrastive_loss(temp_model(views), data.size(0), n_views=n_views, temperature=self.tau)
            new_loss.backward()
            temp_optimiser.step()
            temp_optimiser.zero_grad()

        if new_loss.item() < initial_loss.item():
            self.model = temp_model

    def test_loop(self, base_model, adapt_iters=5):
        assert self.transforms is not None, "transforms required for test-time adaptation"
        device = next(self.model.parameters()).device
        base_model.reset_classifier(10)
        base_model.to(device)
        criterion = nn.CrossEntropyLoss()

        correct = 0
        total = 0
        test_loss = 0

        loop = tqdm(self.test_data, desc="CVP test")
        for data, labels in loop:
            data, labels = data.to(device), labels.to(device)
            self.online_train(data, adapt_iters, self.transforms, self.n_views, device)

            self.model.eval()
            with torch.no_grad():
                output = self.model.backbone(self.model.head(data))
                predictions = output.argmax(dim=1)
                loss = criterion(output, labels)
                test_loss += loss.item() * labels.size(0)
                correct += (predictions == labels).sum().item()
                total += len(labels)

            loop.set_postfix(loss=loss.item(), err=1 - correct / total)

        accuracy = correct / total
        print(f"accuracy : {accuracy}")
        return accuracy
