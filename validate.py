import torch
import torch.nn as nn
import torch.nn.functional as F

def eval_model(model, testloader, criterion, batch_size, device):
    model.eval()
    total = 0
    correct = 0

    persample_loss = 0.0

    with torch.no_grad():
        for i, (data, target) in enumerate(testloader):
            data = data.to(device)
            target = target.to(torch.long)
            target = target.to(device)

            out = model(data)
            
            loss = criterion(out, target)
            persample_loss += loss.item()
            _, pred = torch.max(out, dim=1)

            total += target.size(0)
            correct += (pred == target).sum().item()
    
    accuracy = correct / total
    return (persample_loss/(len(testloader)*batch_size)), accuracy