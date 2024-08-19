import torch
from torchdiffeq import odeint
import numpy as np
 
class Dynamics(torch.nn.Module):
    def __init__(self, dim_state, dim_control):
        super().__init__()
        self.A = torch.nn.Linear(dim_state, dim_state)
        self.B = torch.nn.Linear(dim_control, dim_state)

    def forward(self, x, u):
        dx = self.A(x) + self.B(u)
        return dx
    
class Controller(torch.nn.Module):
    def __init__(self, dim_control, dim_observation):
        super().__init__()
        self.linear = torch.nn.Linear(dim_observation, dim_control)

    def forward(self, x, x_target):
        c = self.linear(x - x_target)
        return c
    
dim_state = 3
dim_control = 3
dim_observation = 3
dynamics = Dynamics(dim_state, dim_control)
control = Controller(dim_control, dim_observation)

x0 = torch.rand(dim_state)

t_final = 1
dt = 0.01
N = int(np.ceil(t_final / dt))
dt = t_final / N
T = torch.tensor(np.linspace(0, t_final, N))

optimizer = torch.optim.Adam(control.parameters(), lr = 0.1)

system = lambda t, x : dynamics(x, control(x, x_target))

for epoch in range(50):

    x_target = torch.rand(dim_state)

    X = odeint(system, x0, T)

    loss = torch.norm(X-x_target) / N

    print(loss)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()