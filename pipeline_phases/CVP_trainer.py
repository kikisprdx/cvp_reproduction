import copy
import os

import torch
import torch.nn as nn

from models.CVP_model import CVP
from utils import contrastive_loss_piecewise


class CVPTrainer:
    def __init__(
        self,
        model: CVP,
        optimiser,
        training_data,
        test_data,
        tau=0.2,
        # transforms=None,
        # n_views=2,
    ):
        self.model = model
        self.optimiser = optimiser
        self.tau = tau

        self.training_data = training_data
        self.test_data = test_data

        # self.transforms = transforms
        # self.n_views = n_views

        self.base_path = "models/cvp/"
        os.makedirs(self.base_path, exist_ok=True)
        self.best_model_path = os.path.join(self.base_path, "cvp")

    def online_train(self, batch_size, num_iterations, lam):
        temp_model = copy.deepcopy(self.model)
        temp_model.train()
        # For every single x, it creates 2
        # possibly augmented version of x
        # does this for every batch
        # -> model evaluates on this
        for batch, (X, y) in enumerate(self.training_data):
            with torch.no_grad():
                initial_loss = contrastive_loss_piecewise(temp_model(X), y, self.tau)
            for t_iteration in range(num_iterations):
                x_adp = temp_model(X)
                new_loss = contrastive_loss_piecewise(x_adp, y, self.tau)
                new_loss.backward()
                self.optimiser.step()
                self.optimiser.zero_grad()
            # pass x through ssl model

            # get contrastive loss
            # update kernel

            # views = torch.cat([self.transforms(X) for _ in range(self.n_views)], dim=0)
            # pred = temp_model(X)
            # loss = temp_model.loss(pred, y, pred, self.tau)
            #
            # loss.backward()

            if batch % 5 == 0:
                loss, current = new_loss.item(), batch * batch_size + len(X)
                print(
                    f"loss: {loss:>7f}  [{current:>5d}/{len(self.training_data.dataset):>5d}]"
                )
            if new_loss.item() < initial_loss.item():
                self.model = temp_model

        # TODO: Double check what optimiser and maybe scheduler is used in the paper
        # torch.save(self.model.state_dict(), self.best_model_path + ".pth")
        # torch.save(self.model, self.best_model_path + "_entire.pth")

    def test_loop(self, adapt_iters=5, lam=0.5):
        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        self.online_train(512, adapt_iters, lam)

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

        results = accuracy # placeholder for now (?)
        return results
