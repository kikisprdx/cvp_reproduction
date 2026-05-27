from models.SVP_model import SVP
import torch
import os

class SVPTrainer:
    def __init__(self, model: SVP, optimiser, training_data, test_data, tau=0.5):
        self.model = model
        self.optimiser = optimiser
        self.tau = tau

        self.training_data = training_data
        self.test_data = test_data
        
        self.base_path = "models/svp/"
        os.mkdir(self.base_path)
        self.best_model_path = os.path.join(self.base_path, "svp")

    def train(self, batch_size):
        self.model.train()

        for batch, (X, y) in enumerate(self.training_data):
            pred = self.model(X)
            loss = self.model.loss(pred, y, pred, self.tau)

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

            if batch % 5 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(
                    f"loss: {loss:>7f}  [{current:>5d}/{len(self.training_data.dataset):>5d}]"
                )
        # TODO: Double check what optimiser and maybe scheduler is used in the paper
        torch.save(self.model.state_dict(), self.best_model_path) # model weights only
        torch.save(self.model, os.path.join(self.best_model_path, "_entire.pth")) # model

    def test_loop(self):
        pass
