import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.adapters.FT_trainer import FTTrainer
from src.adapters.PFT_trainer import PFTTrainer
from src.data.data_loader import get_data_loader_CIFAR10
from src.models.SVP_model import SVPPatch, SVPPad
from src.pipeline_phases.training_pipelines import train_cvpf3, train_cvpr3, train_svp
from src.utils import contrastive_loss, ssl_transform


def training_phase_SSL(base_model, ssl_model, train_loader, epochs=200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssl_model = ssl_model.to(device)
    base_model = base_model.to(device)

    # Optimizer and Scheduler setup as in the paper
    optimizer = optim.AdamW(ssl_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ssl_model_weights_path = "results/ssl_weights.pth"
    ssl_full_model_path = "results/ssl_best_model.pth"

    best_loss = float("inf")

    ssl_model.train()
    base_model.eval()

    # freeze the base model aka "backbone model"
    for param in base_model.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for inputs, _ in loop:  # SSL label free so use same sample
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
                pooled_features = raw_features.mean(
                    dim=(2, 3)
                )  # global average pooling

            # pass features through the SSL MLP model
            ssl_outputs = ssl_model(pooled_features)

            # calculate contrastive loss
            loss = contrastive_loss(
                ssl_outputs, current_batch_size, n_views=3, temperature=0.2
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        average_loss = running_loss / len(train_loader)

        print(
            f"epoch {epoch + 1}/{epochs} | avg Loss: {average_loss:.4f} | LR: {current_lr:.6f}"
        )

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(ssl_model.state_dict(), ssl_model_weights_path)
            torch.save(ssl_model, ssl_full_model_path)
            print(">> Saved new best SSL model!")


def testing_phase_standard(base_model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    base_model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    total = 0

    loop = tqdm(test_loader, desc="baseline test")
    with torch.no_grad():
        for data, labels, *_ in loop:
            data, labels = data.to(device), labels.to(device)
            output = base_model(data)  # model already outputs the raw logits
            predictions = output.argmax(
                dim=1
            )  # this step is bascially the classifer in appendix 7.1
            loss = criterion(output, labels)

            test_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += len(labels)

            loop.set_postfix(loss=loss.item(), err=1 - correct / total)

    accuracy = correct / total
    print(f"accuracy : {accuracy}")

    # make summary structure to contain results and return them
    results = accuracy  # placeholder for now (?)
    return results


def testing_phase_prompting(base_model, ssl_model, test_loader, method, adapt_iters=5):
    print(f"Preparing test-time adaptation for {method}!\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model.to(device)
    base_model.eval()
    ssl_model.to(device)
    train_data = get_data_loader_CIFAR10(batch_size=512)

    if method == "CVP-F3":
        trainer = train_cvpf3(ssl_model, base_model, train_data, test_loader)
    elif method == "CVP-R3":
        trainer = train_cvpr3(ssl_model, base_model, train_data, test_loader)
    elif method == "SVP-Patch":
        svp_model = SVPPatch(base_model, ssl_model)
        svp_model.to(device)
        optimiser = torch.optim.SGD(svp_model.parameters(), lr=2 / 255)
        trainer = train_svp(32, svp_model, optimiser, train_data, test_loader)
    elif method == "SVP-Pad":
        svp_model = SVPPad(base_model, ssl_model)
        svp_model.to(device)
        optimiser = torch.optim.SGD(svp_model.parameters(), lr=2 / 255)
        trainer = train_svp(32, svp_model, optimiser, train_data, test_loader)
    elif method == "FT":
        trainer = FTTrainer(
            model=ssl_model, test_data=test_loader, transforms=ssl_transform, n_views=3
        )
    elif method == "PFT":
        trainer = PFTTrainer(
            model=ssl_model, test_data=test_loader, transforms=ssl_transform, n_views=3
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    accuracy = trainer.test_loop(
        base_model, adapt_iters=adapt_iters
    )  # maybe rename function to adaptation_run or sth

    return accuracy
