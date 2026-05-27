
from models.SVP_model import SVP


class SVPTrainer:
    def __init__(self, model: SVP, optimiser, tau=0.5):
        self.model = model
        self.optimiser = optimiser
        self.tau = tau

        self.training_data = None
        self.test_data = None

    def train(self):
        self.model.train()

        for batch, (X, y) in enumerate(self.training_data):
            pred = self.model(X)
            loss = self.model.loss(X, y, pred, self.tau)

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self):
        pass
