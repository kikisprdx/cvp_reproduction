import os

import torch
import torch.nn as nn
from tqdm import tqdm

from models.SVP_model import SVP
from utils import contrastive_loss


class SVPTrainer:
    def __init__(
        self,
        model: SVP,
        optimiser,
        training_data,
        test_data,
        tau=0.2,
        transforms=None,
        n_views=2,
        scheduler=None,
    ):
        self.model = model
        self.optimiser = optimiser
        self.tau = tau
        self.scheduler = scheduler

        self.training_data = training_data
        self.test_data = test_data

        self.transforms = transforms
        self.n_views = n_views

        self.base_path = "models/svp/"
        os.makedirs(self.base_path, exist_ok=True)
        self.best_model_path = os.path.join(self.base_path, "svp")

    def train(self, batch_size):
        assert self.transforms is not None, "transforms required for multi-view augmentation"
        self.model.train()
        self.model.backbone.eval()

        # SVP:
        # For every single x, it creates 2
        # possibly augmented version of x
        # does this for every batch
        # -> model evaluates on this
        loop = tqdm(self.training_data, desc="SVP train")
        for X, y in loop:
            # Multi view augmentation
            views = torch.cat([self.transforms(X) for _ in range(self.n_views)], dim=0)
            pred = self.model(views)
            loss = contrastive_loss(pred, X.size(0), n_views=self.n_views, temperature=self.tau)

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            loop.set_postfix(loss=loss.item())
        if self.scheduler is not None:
            self.scheduler.step()
        torch.save(self.model.state_dict(), self.best_model_path + ".pth")
        torch.save(self.model, self.best_model_path + "_entire.pth")

    def test_loop(self, base_model, adapt_iters=5):
        assert self.transforms is not None, "transforms required for test-time adaptation"
        device = self.model.prompt.device
        base_model.reset_classifier(10)
        base_model.to(device)
        criterion = nn.CrossEntropyLoss()

        correct = 0
        total = 0
        test_loss = 0

        loop = tqdm(self.test_data, desc="SVP test")
        for data, labels in loop:
            data, labels = data.to(device), labels.to(device)
            self.model.train()
            self.model.backbone.eval()
            for _ in range(adapt_iters):
                views = torch.cat([self.transforms(data.cpu()) for _ in range(self.n_views)], dim=0).to(device)
                pred = self.model(views)
                loss = contrastive_loss(pred, data.size(0), n_views=self.n_views, temperature=self.tau)
                loss.backward()
                self.optimiser.step()
                self.optimiser.zero_grad()

            self.model.eval()
            base_model.eval()
            with torch.no_grad():
                output = self.model.backbone(data + self.model.prompt)
                predictions = output.argmax(dim=1)
                loss = criterion(output, labels)
                test_loss += loss.item() * labels.size(0)
                correct += (predictions == labels).sum().item()
                total += len(labels)

            loop.set_postfix(loss=loss.item(), err=1 - correct / total)

        accuracy = correct / total
        print(f"accuracy : {accuracy}")
        return accuracy
