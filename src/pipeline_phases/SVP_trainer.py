import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.SVP_model import SVPPatch, SVPPad
from src.utils import contrastive_loss


class SVPTrainer:
    """Test-time trainer for SVP: adapts a shared visual prompt via contrastive SSL."""

    def __init__(
        self,
        model,
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

    def test_loop(self, base_model, adapt_iters=5, eps_v=8/255):
        """Evaluate SVP accuracy on test_data with per-batch ℓ₂-constrained prompt adaptation."""
        assert self.transforms is not None, "transforms required for test-time adaptation"
        device = self.model.prompt.device
        base_model.to(device)
        criterion = nn.CrossEntropyLoss()

        correct = 0
        total = 0
        test_loss = 0

        loop = tqdm(self.test_data, desc="SVP test")
        for data, labels, *_ in loop:
            data, labels = data.to(device), labels.to(device)

            self.model.prompt.data.zero_()
            self.optimiser.state.clear()

            self.model.train()
            self.model.backbone.eval()
            for _ in range(adapt_iters):
                views = torch.cat([self.transforms(data) for _ in range(self.n_views)], dim=0)
                pred = self.model(views)
                loss = contrastive_loss(pred, data.size(0), n_views=self.n_views, temperature=self.tau)
                loss.backward()
                self.optimiser.step()
                self.optimiser.zero_grad()
                norm = self.model.prompt.data.norm()
                if norm > eps_v:
                    self.model.prompt.data.mul_(eps_v / norm)

            self.model.eval()
            base_model.eval()
            with torch.no_grad():
                output = self.model.backbone(self.model.apply_prompt(data))
                predictions = output.argmax(dim=1)
                loss = criterion(output, labels)
                test_loss += loss.item() * labels.size(0)
                correct += (predictions == labels).sum().item()
                total += len(labels)

            loop.set_postfix(loss=loss.item(), err=1 - correct / total)

        accuracy = correct / total
        print(f"accuracy : {accuracy}")
        return accuracy
