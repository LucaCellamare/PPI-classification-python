import tqdm
import torch
from torch import nn

'''

STANDARD TRAINING ROUTINE FOR GCN AND SAGE

'''


def train_model(model, X, A,labels, idx_train, epochs, lr, wd):
    labels = torch.FloatTensor(labels)
    X = torch.FloatTensor(X)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in tqdm.trange(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X, A)
        loss = nn.BCEWithLogitsLoss()  # WE CHOSE BINARY CROSS ENTROPY AS COST FUNCTION
        loss = loss(pred[idx_train], labels[idx_train])  # CALCULATING THE LOSS OVER THE TRAINING SET
        loss.backward()  # BACKPROPAGATION
        optimizer.step()
    return pred






