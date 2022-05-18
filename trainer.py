import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, trainloader, learning_rate, num_epochs, batch_size, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    running_loss = 0.0
    loss_history = []
    num_batches = len(trainloader)

    for epoch in range(num_epochs):
        for i, (data_batch, label_batch) in enumerate(trainloader):
            x = data_batch.to(device)
            # one-hot encoding of labels
            label_batch = F.one_hot(label_batch.to(torch.int64))
            label_batch = label_batch.to(torch.float32)
            y = label_batch.to(device)

            out = model(x)
            # print(out)
            loss = criterion(out, y)
            # batch_size = data_batch.size(0)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch%10==0:
                print(f'Epoch [{epoch}/{num_epochs}], Step[{i+1}/{num_batches}], Loss: {loss.item():.4f}') # have to include the step loss

        loss_history.append(running_loss / (num_batches*batch_size)) # average per-sample loss per epoch
        running_loss = 0.0

    return loss_history

