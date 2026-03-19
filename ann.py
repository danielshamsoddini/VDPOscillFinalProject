import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from util import BestModel

#a simple feedforward neural network with variable depth and width, tanh activation
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, depth=2):
        super().__init__()
        nnet_structure = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth-1):
            nnet_structure.extend([nn.Linear(width, width), nn.Tanh()])
        nnet_structure.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*nnet_structure)

    def forward(self, x):
        return self.net(x)

#A training wrapper for the feedforward neural network, pass in device because of GPU
def train_ffnn(time_interval, x_points, epochs=1000, depth=6, width=128, lr=1e-3, device = "cpu"):
    
    model = MLP(1, 1, width=width, depth=depth).to(device)
    best_model = BestModel()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    time_interval = torch.tensor(time_interval, dtype=torch.float32).view(-1, 1)
    x_points = torch.tensor(x_points, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(TensorDataset(time_interval, x_points), batch_size=256, shuffle=True)

    model.train()
    loss_history = []
    for a in range(epochs):
        current_loss = 0.0
        for time_batch, x_batch in loader:
            time_batch, x_batch = time_batch.to(device), x_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(time_batch), x_batch)
            loss.backward()
            optimizer.step()
            current_loss += loss.item() * time_batch.shape[0]
        
        loss_history.append(current_loss / len(loader.dataset))
        best_model.update(model, loss_history[-1])
        if a % 100 == 0:
            print("Epoch", a, "Loss", loss_history[-1])

    model = best_model.load_best(model)
    return model, loss_history

