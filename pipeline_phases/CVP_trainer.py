import os

import torch
import torch.nn as nn
from tqdm import tqdm

from models.CVP_model import CVPF3
from utils import contrastive_loss


class CVPTrainer:
    def __init__(
        self,
        model: CVPF3,
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

    def _ssl_loss(self, data, n_views):
        adapted = self.model.head(data)
        views = torch.cat([self.transforms(adapted) for _ in range(n_views)], dim=0)
        features = self.model.backbone.forward_features(views).mean(dim=(2, 3))
        return contrastive_loss(self.model.ssl_model(features), data.size(0), n_views=n_views, temperature=self.tau)

    def online_train(self, data, num_iterations, n_views):
        initial_kernel = self.model.head.conv.weight.data.clone()
        initial_bias = self.model.head.conv.bias.data.clone()
        initial_lam = self.model.head.lam.data.clone()

        self.model.train()
        self.model.backbone.eval()
        self.model.ssl_model.eval()

        with torch.no_grad():
            views = torch.cat([self.transforms(data) for _ in range(n_views)], dim=0)
            features = self.model.backbone.forward_features(views).mean(dim=(2, 3))
            initial_loss = contrastive_loss(self.model.ssl_model(features), data.size(0), n_views=n_views, temperature=self.tau)

        for _ in range(num_iterations):
            self.optimiser.zero_grad()
            loss = self._ssl_loss(data, n_views)
            loss.backward()
            self.optimiser.step()
            self.model.head.lam.data.clamp_(0.5, 3.0)

        with torch.no_grad():
            final_loss = self._ssl_loss(data, n_views)

        if final_loss.item() >= initial_loss.item():
            self.model.head.conv.weight.data.copy_(initial_kernel)
            self.model.head.conv.bias.data.copy_(initial_bias)
            self.model.head.lam.data.copy_(initial_lam)

    def test_loop(self, base_model, adapt_iters=5):
        assert self.transforms is not None, "transforms required for test-time adaptation"
        device = next(self.model.parameters()).device
        base_model.to(device)
        criterion = nn.CrossEntropyLoss()

        orig_kernel = self.model.head.conv.weight.data.clone()
        orig_bias = self.model.head.conv.bias.data.clone()
        orig_lam = self.model.head.lam.data.clone()

        correct = 0
        total = 0
        test_loss = 0

        loop = tqdm(self.test_data, desc="CVP test")
        for data, labels in loop:
            data, labels = data.to(device), labels.to(device)
            self.model.head.conv.weight.data.copy_(orig_kernel)
            self.model.head.conv.bias.data.copy_(orig_bias)
            self.model.head.lam.data.copy_(orig_lam)
            self.online_train(data, adapt_iters, self.n_views)

            self.model.eval()
            with torch.no_grad():
                output = self.model.backbone(self.model.head(data))
                predictions = output.argmax(dim=1)
                loss = criterion(output, labels)
                test_loss += loss.item() * labels.size(0)
                correct += (predictions == labels).sum().item()
                total += len(labels)

            loop.set_postfix(loss=loss.item(), err=1 - correct / total, acc=correct / total)

        accuracy = correct / total
        print(f"accuracy : {accuracy}")
        return accuracy
