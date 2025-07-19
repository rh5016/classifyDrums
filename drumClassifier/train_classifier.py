import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# load features
X = np.load("embs.npy") 
y = np.load("lbls.npy")

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

dataset = TensorDataset(X,y)
n_val   = int(len(dataset)*0.2)
n_train = len(dataset)-n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16)

#Small classifier
class Head(nn.Module):
    def __init__(self, dim, n_classes=3):
        super().__init__()
        self.lin = nn.Linear(dim, n_classes)
    def forward(self, x):
        return self.lin(x)

model = Head(X.shape[1]).to("cuda")
opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
crit  = nn.CrossEntropyLoss()

#training
for epoch in range(10):
    model.train()
    tot, acc = 0, 0
    for xb,yb in train_loader:
        xb, yb = xb.cuda(), yb.cuda()
        logits = model(xb)
        loss   = crit(logits,yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += yb.size(0)
        acc += (logits.argmax(1)==yb).sum().item()
    print(f"epoch {epoch} : train acc {acc/tot:.3f}")

    # val
    model.eval()
    tot, acc = 0, 0
    with torch.no_grad():
      for xb,yb in val_loader:
        xb,yb = xb.cuda(), yb.cuda()
        pred  = model(xb).argmax(1)
        tot  += yb.size(0)
        acc  += (pred==yb).sum().item()
    print(f"val acc {acc/tot:.3f}")
