import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from util import BestModel
#This code is inspired by the diffusion PINN example from class, however mine solves for the Van der Pol oscillators, as such it has no upper boundary condition per-se

#generic feedforward neural network, identical to nn but with xavier initialization
class PINN(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, depth=6):
        super().__init__()
        nnet_structure = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth-1):
            nnet_structure.extend([nn.Linear(width, width), nn.Tanh()])
        nnet_structure.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*nnet_structure)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)

#Function that trains the PINN, based on the in-class example, modified for ODEs, the Van der Pol oscillator, and to test my collocation modes
def train_pinn(t0, t_max, initial_state, mu, A=0.0, omega=1.0, forced=False, time_interval=None, x_points=None, y_points=None, epochs=2000, num_collocation=3000, w_ode=1.0, w_ic=10.0, w_data=10.0, lr=1e-3, width=128, depth=6, device="cpu", print_interval=100, collocation_mode="100"):

    rng = np.random.default_rng(1)
    
    model = PINN(in_dim=1, out_dim=2, width=width, depth=depth).to(device)
    best_model = BestModel()

    #convert everything to tensors, otherwise pytorch throws an error!
    initial_x = torch.tensor([[float(initial_state[0])]], dtype=torch.float32, device=device)
    initial_y = torch.tensor([[float(initial_state[1])]], dtype=torch.float32, device=device)
    start_time = torch.tensor([[float(t0)]], dtype=torch.float32, device=device)
    mu_tensor = torch.tensor(float(mu), dtype=torch.float32, device=device)
    A_tensor = torch.tensor(float(A), dtype=torch.float32, device=device)
    omega_tensor = torch.tensor(float(omega), dtype=torch.float32, device=device)

    sampled_time_tensor = torch.tensor(time_interval, dtype=torch.float32, device=device).view(-1, 1)
    sampled_x_tensor = torch.tensor(x_points, dtype=torch.float32, device=device).view(-1, 1)
    sampled_y_tensor = torch.tensor(y_points, dtype=torch.float32, device=device).view(-1, 1)

    def get_collocation_points():
        if collocation_mode == "bounded":
            sampled_time_points = rng.uniform(t0, t_max, (num_collocation, 1))
        elif collocation_mode == "100":
            sampled_time_points = rng.uniform(0.0, 100.0, (num_collocation, 1))
        elif collocation_mode == "1000":
            sampled_time_points = rng.uniform(0.0, 1000.0, (num_collocation, 1))
        else:
            raise ValueError(
                "collocation_mode should either be bounded, 100, or 1000"
            )

        t_f = torch.tensor(sampled_time_points, dtype=torch.float32, device=device)
        t_f.requires_grad_(True)
        return t_f

    def loss_function():
        t_f = get_collocation_points()
        predicted_collocation_states = model(t_f)
        x_p, y_p = predicted_collocation_states[:, 0:1], predicted_collocation_states[:, 1:2]
        x_residual = torch.autograd.grad(x_p, t_f, grad_outputs=torch.ones_like(x_p), create_graph=True, retain_graph=True,)[0] - y_p
        forcing = A_tensor * torch.sin(omega_tensor * t_f) if forced else 0.0
        y_residual = torch.autograd.grad(y_p, t_f, grad_outputs=torch.ones_like(y_p), create_graph=True, retain_graph=True,)[0] - (mu_tensor * (1 - x_p ** 2) * y_p - x_p + forcing)
        L_ode = torch.mean(x_residual ** 2) + torch.mean(y_residual ** 2)

        predicted_initial_states = model(start_time)
        L_ic = F.mse_loss(predicted_initial_states[:, 0:1], initial_x) + F.mse_loss(predicted_initial_states[:, 1:2], initial_y)

        predicted_data_states = model(sampled_time_tensor)
        L_data = F.mse_loss(predicted_data_states[:, 0:1], sampled_x_tensor) + F.mse_loss(predicted_data_states[:, 1:2], sampled_y_tensor)

        return w_ode * L_ode + w_ic * L_ic + w_data * L_data

    loss_history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for a in range(epochs):
        optimizer.zero_grad()
        loss = loss_function()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        best_model.update(model, loss_history[-1])
        if a < 10 or a % print_interval == 0:
            print("Epoch #", a, "Loss", loss.item())

    model = best_model.load_best(model)
    return model, loss_history

#helper function to get predictions from the PINN
def pinn_predict(model, time_interval, device="cpu"):
    model.eval()
    time_tensor = torch.tensor(time_interval, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        predicted_states = model(time_tensor).cpu().numpy()
    return predicted_states[:, 0], predicted_states[:, 1]
