from models.SSL_model import SSL_model
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from train_svp import SVPTrainer

# the augmentations specified by the paper 
ssl_transform = T.Compose([
    T.RandomResizedCrop(32, antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=[-90, 90])
])

def contrastive_loss(features, batch_size, n_views=3, temperature=0.1):
    """Implementation of Equation 1 from the paper."""
    # normalize features for cosine similarity
    features = F.normalize(features, dim=1)
    
    # calculate cosine similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temperature

    # mask out self-similarity -> don't want to compare a view to itself
    mask = torch.eye(batch_size * n_views, dtype=torch.bool, device=features.device)
    sim_matrix.masked_fill_(mask, -9e15)

    # labels: y_i,j is a 0-1 vector indicating positive pairs
    # views from the same original image share the same label
    labels = torch.arange(batch_size, device=features.device).repeat(n_views)

    # boolean mask of positive pairs (same label, but not the exact same view)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask

    # compute log probabilities
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

    # calculate the mean log-probability of positive pairs
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)

    # negative expected value
    loss = -pos_log_prob.mean()
    return loss


def training_phase_SSL(base_model, ssl_model, train_loader, epochs=200): # Epochs set to 200 [cite: 652]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_model = ssl_model.to(device)
    base_model = base_model.to(device)
    
    # Optimizer and Scheduler setup as in the paper
    optimizer = optim.AdamW(ssl_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    ssl_model_weights_path = 'models/ssl_weights.pth'
    ssl_full_model_path = 'models/ssl_best_model.pth'

    patience = 5 #idk
    best_loss = np.inf
    counter = 0

    ssl_model.train()
    base_model.eval()
    
    # freeze the base model aka "backbone model"
    for param in base_model.parameters():
        param.requires_grad = False
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, _ in loop: # SSL label free so use same sample
            inputs = inputs.to(device)
            current_batch_size = inputs.size(0)
            
            optimizer.zero_grad()
            
            # 3 distinct augmented views per sample 
            views = [ssl_transform(inputs) for _ in range(3)]
            
            # stack views along the batch dimension -> shape: (3*B, C, H, W)
            stacked_views = torch.cat(views, dim=0)
            
            # extract features before the fully connected layer using timm
            with torch.no_grad():
                raw_features = base_model.forward_features(stacked_views) 
                pooled_features = raw_features.mean(dim=(2, 3)) # global average pooling
            
            # pass features through the SSL MLP model
            ssl_outputs = ssl_model(pooled_features)
            
            # calculate contrastive loss
            loss = contrastive_loss(ssl_outputs, current_batch_size, n_views=3)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        average_loss = running_loss / len(train_loader)
        
        print(f"epoch {epoch+1}/{epochs} | avg Loss: {average_loss:.4f} | LR: {current_lr:.6f}")
        
        if average_loss < best_loss:
            best_loss = average_loss
            counter = 0
            torch.save(ssl_model.state_dict(), ssl_model_weights_path) 
            torch.save(ssl_model, ssl_full_model_path) 
            print(">> Saved new best SSL model!")
        else:
            counter += 1
        
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

'''
def training_phase_SSL(base_model, ssl_model, train_loader, optimizer, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssl_model = ssl_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(ssl_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_model_path = 'models/best_resnet26.pth'
    ssl_model_path = 'models/'

    patience = 3
    best_loss = np.inf
    counter = 0

    simclr_transform = T.Compose([
    T.RandomResizedCrop(32),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.GaussianBlur(kernel_size=3),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

    ssl_model.train()
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    
    for epoch in tqdm(range(epochs)):

        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = base_model(inputs)
            outputs = ssl_model(logits)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        average_loss = running_loss / len(train_loader)
        
        print(f"epoch {epoch+1}/{epochs} - loss: {average_loss:.4f} | LR: {current_lr:.6f}")
        if average_loss < best_loss:
            best_loss = average_loss
            counter = 0

            torch.save(ssl_model.state_dict(), ssl_model_path) # model weights only
            torch.save(ssl_model, 'ssl_best_model.pth') # model
            print(">> Saved new best ssl model!")
        else:
            counter +=1
        
        if counter >= patience:
            break
'''


def testing_phase_standard(base_model, test_loader):

    criterion = nn.CrossEntropyLoss()

    base_model.eval()
    test_loss = 0 
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            output = base_model(data) # model already outputs the raw logits
            #print(output.shape)
            predictions = output.argmax(dim=1) # this step is bascially the classifer in appendix 7.1 
            loss = criterion(output, labels)

            test_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()

            total += len(labels)

    print(total)
    accuracy = correct / total
    print(f"accuracy : {accuracy}")

    # make summary structure to contain results and return them
    results = accuracy # placeholder for now (?)
    return results

def testing_phase_prompting(base_model, ssl_model, test_loader, method='CVP', adapt_iters=5):
    print(f"Preparing test-time adaptation for {method}!\n")

    if method == 'CVP':
        trainer = SVPTrainer( #CVPTrainer
            model=ssl_model, test_data=test_loader, transforms=ssl_transform, n_views=3
        )
    elif method == 'SVP':
        trainer = SVPTrainer(
            model=ssl_model, test_data=test_loader, transforms=ssl_transform, n_views=3
        )
    elif method == 'FT':
        trainer = SVPTrainer( # change to FTTrainer
            model=ssl_model, test_data=test_loader, transforms=ssl_transform, n_views=3
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    accuracy = trainer.test_loop(base_model, adapt_iters=adapt_iters) # maybe rename function to adaptation_run or sth

    return accuracy