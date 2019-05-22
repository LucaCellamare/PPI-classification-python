import os.path as osp
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score
import pandas as pd

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='test')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)
        #self.lin3 = torch.nn.Linear(4 * 256, 200)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)

def vld_loss():
    model.eval()

    total_loss = 0
    for data in val_loader:
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(val_loader.dataset)



def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

train_acc = []
valid_acc = []
train_loss = []
valid_loss = []
vl_loss=0
tr_acc=0
start_time=time.time()

for epoch in range(1,50):
    loss = train()
    tr_acc = test(train_loader)
    vl_loss = vld_loss()
    acc = test(val_loader)

    train_loss.append(loss)
    train_acc.append(tr_acc)
    valid_loss.append(vl_loss)
    valid_acc.append(acc)

    print('Epoch: {:02d}, train_loss: {:.4f},train_F1: {:.4f},valid_loss: {:.4f}, valid_F1: {:.4f}'.format(epoch, loss, tr_acc, vl_loss, acc))

print("execution time= %s sec "%(time.time()-start_time))
df=pd.DataFrame(train_loss)
df.to_csv('train_loss.csv')

df=pd.DataFrame(train_acc)
df.to_csv('train_acc.csv')

df=pd.DataFrame(valid_loss)
df.to_csv('valid_loss.csv')

df=pd.DataFrame(valid_acc)
df.to_csv('valid_acc.csv')

#torch.save(model.state_dict(),'path_to../pytorch_geometric-master/examples/gat-trained/gat_true_train_computer.pt')
