import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from src.utils import contrastive_loss
from src.adapters.base_adapter import BaseTrainer

class PFTTrainer(BaseTrainer):

    def test_loop(self, base_model: nn.Module, adapt_iters: int = 5):
        base_model = base_model.to(self.device)
        
        self.ssl_model.eval() # ssl frozen 
        for param in self.ssl_model.parameters():
            param.requires_grad = False

        correct = 0
        total = 0

        loop = tqdm(self.test_data, desc="Test-Time Adaptation (PFT)")
        
        for inputs, labels, *_ in loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.size(0)

            initial_model_state = copy.deepcopy(base_model.state_dict()) # deep copy to revert like FT method

            base_model.train()
            
            for param in base_model.parameters():
                param.requires_grad = False
                
            # ufreeze only the batchnorm layers
            for module in base_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = True
            
            # give the optimizer the parameters that require gradients -> batchnorm
            trainable_params = filter(lambda p: p.requires_grad, base_model.parameters())
            optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

            for t in range(adapt_iters): # adaptation 
                optimizer.zero_grad()
                
                views = [self.transforms(inputs) for _ in range(self.n_views)]
                stacked_views = torch.cat(views, dim=0)
                
                # extract features and calculate SSL Loss
                raw_features = base_model.forward_features(stacked_views)
                pooled_features = raw_features.mean(dim=(2, 3))
                ssl_outputs = self.ssl_model(pooled_features)
                
                loss = contrastive_loss(ssl_outputs, batch_size, n_views=self.n_views, temperature=0.2)
                
                loss.backward()
                optimizer.step()

            base_model.eval() # now inference
            with torch.no_grad():
                outputs = base_model(inputs)
                predictions = outputs.argmax(dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)


            base_model.load_state_dict(initial_model_state) # reset
            
            loop.set_postfix(acc=f"{correct/total:.4f}")

        accuracy = correct / total
        print(f"Final PFT Accuracy: {accuracy:.4f}")
        return accuracy