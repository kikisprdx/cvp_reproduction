# FIX: missing multi-view augmentation — generate n_views random crops of X and cat before passing to model
# FIX: backbone should stay in model.eval(), only head should be in train mode

import os

import torch
import torch.nn as nn

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
    ):
        self.model = model
        self.optimiser = optimiser
        self.tau = tau

        self.training_data = training_data
        self.test_data = test_data

        self.transforms = transforms
        self.n_views = n_views

        self.base_path = "models/svp/"
        os.makedirs(self.base_path, exist_ok=True)
        self.best_model_path = os.path.join(self.base_path, "svp")

    def train(self, batch_size):
        self.model.train()
        # SVP:
        # For every single x, it creates 2
        # possibly augmented version of x
        # does this for every batch
        # -> model evaluates on this
        for batch, (X, y) in enumerate(self.training_data):
            # Multi view augmentation
            views = torch.cat([self.transforms(X) for _ in range(self.n_views)], dim=0)
            pred = self.model(views)
            loss = contrastive_loss(pred, X.size(0), n_views=self.n_views)

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            if batch % 5 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(
                    f"loss: {loss:>7f}  [{current:>5d}/{len(self.training_data.dataset):>5d}]"
                )
        # TODO: Double check what optimiser and maybe scheduler is used in the paper
        torch.save(self.model.state_dict(), self.best_model_path + ".pth")
        torch.save(self.model, self.best_model_path + "_entire.pth")

    def test_loop(self, base_model, adapt_iters=5):
        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        # TODO: TQDM so i know what's going on and current stats
        with torch.no_grad():
            for data, labels in self.test_data:
                output = self.model(data)
                predictions = output.argmax(dim=1)
                loss = criterion(output, labels)

                test_loss += loss.item() * labels.size(0)
                correct += (predictions == labels).sum().item()

                total += len(labels)

        print(total)
        accuracy = correct / total
        print(f"accuracy : {accuracy}")

        results = accuracy  # placeholder for now (?)
        return results
