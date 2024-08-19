import torch
import torch.nn.functional as F
import parse_data
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt

class Table(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.values = torch.zeros( ( dim1.shape[0], dim2.shape[0] ) , requires_grad=True)
        self.values.retain_grad()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, point):
        weights1 = 1 / ( (self.dim1 - point[0])**2 + 1 )
        weights1 = weights1 / torch.sum(weights1)
        weights2 = 1 / ( (self.dim2 - point[1])**2 + 1 )
        weights2 = weights2 / torch.sum(weights2)

        mask = torch.outer(weights1, weights2)
        bla = torch.sum(mask * self.values)
        return bla
    
class Environment0(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.c0 = torch.tensor(0.35, requires_grad=True)
        self.c1 = torch.tensor(0.9, requires_grad=True)
        self.area = torch.tensor(torch.pi / 4 * (0.075)**2, requires_grad=True)
        self.rho = torch.tensor(1.225, requires_grad=True)
        self.mass = torch.tensor(2.166, requires_grad=True)

    def forward(self, state: tuple[torch.tensor], action: torch.tensor) -> torch.tensor:
        altitude, velocity = state
        force = -(0.5 * self.rho * velocity**2) * self.area * (self.c0 * action + self.c1 * (1 - action))
        return (velocity, force / self.mass - 9.81)

class Policy0(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2,1)
        self.linear.weight.data = torch.tensor([[0., 0.]])
        self.linear.bias.data = torch.tensor([0.5])
    
    def forward(self, state: tuple[torch.tensor]):
        action = self.linear(torch.cat(state))
        #action = torch.clamp(action, 0, 1)
        return action

class Policy1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.table = Table(torch.linspace(0, 1000, 10), torch.linspace(0, 300, 10))
    
    def forward(self, state: tuple[torch.tensor]):
        action = self.table(state)
        return action

class Policy2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2,10)
        self.linear2 = torch.nn.Linear(10,1)

    
    def forward(self, state: tuple[torch.tensor]):
        action = self.linear1(torch.cat(state))
        action = F.relu(action)
        action = self.linear2(action)
        action = torch.clamp(action, 0, 1)
        return action

class System0(torch.nn.Module):

    def __init__(self, policy, environment):
        super().__init__()
        self.policy = policy
        self.environment = environment

    def forward(self, t, state: tuple[torch.tensor]):
        with torch.enable_grad():
            action = self.policy(state)
            d_state = self.environment(state, action)
            return d_state

def main():

    policy =  Policy2()
    environment = Environment0()
    sys = System0(policy, environment)

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    ave_loss = 0

    for epoch in range(300):

        alt_0 = torch.rand(1) * 100
        vel_0 = torch.rand(1) * 100 + 50
        state_0 = (alt_0, vel_0)

        t = torch.linspace(0.0, 10.0, 1000)
        target_altitude = 320

        optimizer.zero_grad()

        output = odeint(sys, state_0, t, method='rk4')
        apogee = torch.max(output[0])
        loss = (target_altitude - apogee)**2
        loss.backward()
        optimizer.step()

        ave_loss += loss
        if (epoch + 1) % 30 == 0:
            print(ave_loss / 30)
            ave_loss = 0

    plt.plot(output[:,0].detach().numpy())
    plt.show()


if __name__ == "__main__":
    main()