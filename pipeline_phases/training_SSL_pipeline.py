from models.SSL_model import SSL_model
import torch
import torch.nn as nn





def training_phase_SSL(base_model, ssl_model, train_loader, optimizer, epoch=100):
    pass

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

def testing_phase_prompting(base_model, ssl_model, test_loader, method='CVP'):
    pass