import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils import contrastive_loss
from src.adapters.base_adapter import BaseTrainer

class PFTTrainer(BaseTrainer):

    def test_loop(self, base_model: nn.Module, adapt_iters: int = 5):
        """Adapt only BatchNorm2d parameters per batch via contrastive SSL loss, then infer."""
        base_model = base_model.to(self.device)

        self.ssl_model.eval()
        for param in self.ssl_model.parameters():
            param.requires_grad = False

        bn_params = [p for module in base_model.modules()
                     if isinstance(module, nn.BatchNorm2d)
                     for p in module.parameters()]

        scaler = torch.amp.GradScaler("cuda")
        correct = 0
        total = 0

        loop = tqdm(self.test_data, desc="Test-Time Adaptation (PFT)")

        for inputs, labels, *_ in loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.size(0)

            bn_state = {}
            for mod_name, module in base_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    for k, v in module.state_dict().items():
                        bn_state[f"{mod_name}.{k}"] = v.clone()

            base_model.train()
            optimizer = torch.optim.Adam(bn_params, lr=1e-4)

            for _ in range(adapt_iters):
                base_model.zero_grad()
                views = [self.transforms(inputs) for _ in range(self.n_views)]
                stacked_views = torch.cat(views, dim=0)

                with torch.autocast(device_type="cuda"):
                    raw_features = base_model.forward_features(stacked_views)
                    pooled_features = raw_features.mean(dim=(2, 3))
                    ssl_outputs = self.ssl_model(pooled_features)
                    loss = contrastive_loss(ssl_outputs, batch_size, n_views=self.n_views, temperature=0.2)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            base_model.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                outputs = base_model(inputs)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            for mod_name, module in base_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.load_state_dict({k: bn_state[f"{mod_name}.{k}"] for k in module.state_dict()})

            loop.set_postfix(acc=f"{correct/total:.4f}")

        accuracy = correct / total
        print(f"Final PFT Accuracy: {accuracy:.4f}")
        return accuracy
