import torch
import torch.nn as nn
import torch.optim as optim

import time
from util import BestModel

#My neural ode, it has 2 hidden layers and a tanh activation function
class NeuralODE(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        #input is x, x', and t
        #change num layers or act as needed
        self.vector_field = nn.Sequential(
            nn.Linear(3, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 2)
        )

    def forward(self, h, t=None):
        is_unbatched = (h.dim() == 1)
        if is_unbatched:
            h = h.unsqueeze(0)
            
        #add time as a feature if forced VDP oscillator
        t = 0.0 if t is None else t
        t_tensor = torch.as_tensor(t, dtype=h.dtype, device=h.device)
        t_feat = t_tensor.view(-1, 1).expand(h.shape[0], 1)
        
        out = self.vector_field(torch.cat([h, t_feat], dim=-1))
        return out.squeeze(0) if is_unbatched else out

#My euler solver for the neural ode, it takes the neural ode as a function and solves for the states
def euler_for_node(func, z_0, dt, steps, t0=0.0):
    z_states = []
    z = z_0
    t = float(t0)
    for _ in range(steps):
        z_states.append(z)
        z = z + dt * func(z, t)
        t += dt
    return torch.stack(z_states, dim=0)


#Training wrapper for neural ode, pass in device because of GPU, this one returns the node, losses, and states for convenience
def train_neural_ode(trajectory_data, dt, device="cpu", epochs=5000, width=128, lr=1e-3, t0=0.0):
    node = NeuralODE(width=width).to(device)
    best_model = BestModel()

    optimizer = optim.AdamW(node.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    z_states = torch.tensor(trajectory_data, dtype=torch.float32, device=device)
    z0_torch = z_states[0]
    steps = z_states.shape[0]
    timetime = time.time()
    node.train()

    loss_history = []
    for a in range(epochs):
        optimizer.zero_grad()
        predicted_z_states = euler_for_node(node, z0_torch, dt, steps, t0=t0)
        loss = loss_fn(predicted_z_states, z_states)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        best_model.update(node, loss_history[-1])

        if a % 100 == 0:
            print("Epoch #", a, "Loss", loss.item())
            print("time:", time.time() - timetime)
            timetime = time.time()

    node = best_model.load_best(node)
    node.eval()

    with torch.no_grad():
        predicted_z_states = euler_for_node(node, z0_torch, dt, steps, t0=t0).cpu().numpy()

    return node, predicted_z_states, loss_history
