import torch
import torch.nn.functional as F
import copy
from tqdm import tqdm
from utils import contrastive_loss
from adapters.base_adapter import BaseTrainer

class FTTrainer(BaseTrainer):

    def test_loop(self, base_model, adapt_iters=5):
        base_model = base_model.to(self.device)
        
        #SSL model is frozen during test-time adaptation
        self.ssl_model.eval()
        for param in self.ssl_model.parameters():
            param.requires_grad = False

        correct = 0
        total = 0

        loop = tqdm(self.test_data, desc="Test-Time Adaptation (FT)")
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.size(0)

            # deepcopy the state per batch so always revert back to the exact same pre-trained starting point
            initial_model_state = copy.deepcopy(base_model.state_dict())

            # unfreeze the base model for fine-tuning
            base_model.train()
            for param in base_model.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-4)

            #adaptation time
            for t in range(adapt_iters):
                optimizer.zero_grad()
                
                # generate views of the UNALTERED corrupted image 
                views = [self.transforms(inputs) for _ in range(self.n_views)]
                stacked_views = torch.cat(views, dim=0)
                
                # extract features and calculate SSL Loss
                raw_features = base_model.forward_features(stacked_views)
                pooled_features = raw_features.mean(dim=(2, 3))
                ssl_outputs = self.ssl_model(pooled_features)
                
                loss = contrastive_loss(ssl_outputs, batch_size, n_views=self.n_views)
                
                loss.backward()
                optimizer.step()
            
            #back to eval mode for deterministic inference
            base_model.eval()
            with torch.no_grad():
                # get predictions on the original corrupted inputs
                outputs = base_model(inputs)
                predictions = outputs.argmax(dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            # remove the fine-tuning for the next batch (???)
            base_model.load_state_dict(initial_model_state)
            
            loop.set_postfix(acc=f"{correct/total:.4f}")

        accuracy = correct / total
        print(f"Final FT Accuracy: {accuracy:.4f}")
        return accuracy