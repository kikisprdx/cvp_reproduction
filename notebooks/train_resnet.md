The paper seems to train their models instead of using pretrained models to fine tune on. 
# %%[jupynvim:cell-sep]
import timm
import torch
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

import torch.optim as optim
import torch.nn as nn


print(Path.cwd().parent.parent)

# %%[jupynvim:cell-sep]

sys.path.insert(0, str(Path.cwd().parent))  # project root
from data.data_loader import get_data_loader_CIFAR10

# %%[jupynvim:cell-sep]
model = timm.create_model('resnet26', pretrained=False, num_classes=10)

train_loader = get_data_loader_CIFAR10(batch_size=64, train=True, notebook=True)
#dataset.shape

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
epochs = 100
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# %%[jupynvim:cell-sep]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

patience = 3
best_loss = np.inf
counter = 0

best_model_path = "../models/best_resnet26.pth"
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
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

        torch.save(model.state_dict(), best_model_path) # model weights only
        torch.save(model, 'entire_resnet26_model_cifar.pth') # model
        print(">> Saved new best model!")
    else:
        counter +=1
    
    if counter >= patience:
        break

model.load_state_dict(torch.load(best_model_path))
# kiki pls train !!
# %%[jupynvim:cell-sep]

