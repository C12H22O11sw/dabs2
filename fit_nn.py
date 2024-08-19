from parse_data import APCPropeller, Airframe

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class PropAPC1(torch.nn.Module):
    def __init__(self, order):
        super().__init__()
        
        self.order = order
        self.power_combinations = (self.order + 1) * (self.order + 2) // 2
        self.linear = torch.nn.Linear(self.power_combinations , 6)
        torch.nn.init.zeros_(self.linear.weight) 


    def forward(self, x):

        if len(x.shape) == 1:
            x = torch.reshape(x, (1, x.shape[0]))

        # create powers of elements of x 
        x_powers = torch.empty((x.shape[0], self.power_combinations))
        index = 0
        for i in range(0, self.order + 1):
            for j in range(0, self.order - i + 1):
                x_powers[:,index] = (torch.pow(x[:,0] / 10000.0, i) * torch.pow(x[:,1] / 100, j))
                index += 1

        wrench = self.linear(x_powers)

        return wrench
   
    
class PropAPC0(torch.nn.Module):
    def __init__(self):
        super().__init__()

        hl1_dim = 1024
        hl2_dim = 1024
        hl3_dim = 1024

        self.linear1 = torch.nn.Linear(2, hl1_dim)
        self.linear2 = torch.nn.Linear(hl1_dim, hl2_dim)
        self.linear3 = torch.nn.Linear(hl2_dim, hl3_dim)
        self.linear4 = torch.nn.Linear(hl3_dim, 6)

    def forward(self, x):

        x = x.to(torch.float)
        if len(x.shape) == 1:
            x = torch.reshape(x, (1, x.shape[0]))

        x[:,0] /= 36000
        x[:,1] /= 200
        
        #x = torch.log(x)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.linear4(x)

        return x


class PropAPC2(torch.nn.Module):
    def __init__(self, max_rpm, max_vel, num):
        super().__init__()

        self.hilbert_velocity = Hilbertizer(0, max_vel, num)
        self.hilbert_rpm = Hilbertizer(0, max_rpm, num)

        hl1_dim = 64
        hl2_dim = 128
        hl3_dim = 512

        self.linear1 = torch.nn.Linear(2 * num, hl1_dim)
        self.linear2 = torch.nn.Linear(hl1_dim, hl2_dim)
        self.linear3 = torch.nn.Linear(hl2_dim, hl3_dim)
        self.linear4 = torch.nn.Linear(hl3_dim, 6)

    def forward(self, x):

        x = x.to(torch.float)
        if len(x.shape) == 1:
            x = torch.reshape(x, (1, x.shape[0]))

        x_rpm = self.hilbert_rpm(x[:,0])
        x_vel = self.hilbert_velocity(x[:,1])

        x = torch.cat((x_rpm, x_vel), dim=1)
        
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.linear4(x)
        return x  

def train(model, propeller, max_rpm=30000, max_velocity=100, epochs=1024, batch_size = 1024):

    criterion = torch.nn.MSELoss()

    lowest_loss = np.inf
    best_model = None

    for epoch in range(epochs):

        if epoch % 64 == 0:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

        x = torch.rand((batch_size, 2)) * torch.tensor([max_rpm, max_velocity])
        y = torch.tensor(propeller.get_wrench(x[:,0], x[:,1]), dtype=torch.float)

        y_pred = model.forward(x)

        loss = criterion(y, y_pred).to(torch.float)
        #if loss < lowest_loss:
        #    best_model = torch.clone(model)

        print(loss)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        #model = best_model

    return model


def compare_thrust_curve(model, prop: APCPropeller, 
                         rpm: int|float=15000, max_velocity: int|float=100):

    T = np.linspace(0, max_velocity, 500)
    true_thrust = []
    pred_thrust = []

    for velocity in T:
        true_thrust.append(prop.get_thrust(rpm, velocity))
        pred_thrust.append(model.forward(torch.tensor([rpm, velocity]))[0][0].detach())

    plt.plot(T, true_thrust, label='true')
    plt.plot(T, pred_thrust, label='pred')
    plt.legend(loc='best')
    plt.show()


class RegularGridInterpolatorPyTorch(torch.nn.Module):
    def __init__(self, shape, start, end):
        super().__init__()

        assert len(shape) == len(start), len(start) == len(end)

        self.grid = torch.zeros(shape)
        self.start = torch.asarray(start)
        self.end = torch.asarray(end)
        self.shape = torch.asarray(shape)
        self.spacing = (self.end - self.start) / torch.asarray(shape)

    def forward(self, x: torch.tensor) -> torch.tensor:
        idx = (x - self.start) / self.spacing
        idx = torch.clamp(idx, 0 * self.shape, self.shape - 1)
        idx = list(idx)

        #self.grid[]

        print()



class Hilbertizer(torch.nn.Module):

    def __init__(self, begin, end, num):
        super().__init__()
        self.begin = begin
        self.end = end
        self.num = num
        self.spacing = (end - begin) / (num - 1)
        self.points = torch.linspace(begin, end, num)

    def forward(self, x):
        weights = torch.tile(self.points, (x.shape[0], 1))
        x = torch.tile(x, (weights.shape[1], 1)).T
        weights = F.relu(torch.abs(weights - x) / self.spacing)
        return weights
                

#points, values = parse_prop_apc_list('data/PER3_5x75E.dat')
#points = list(map(torch.tensor, points))
#values = list(map(torch.tensor, values))

model = RegularGridInterpolatorPyTorch( (100, 100, 100), (0,1,2), (10, 11, 12))
model(torch.tensor([2,4,8]))

#bla = LinearNDInterpolaterModule(points, values)
prop = APCPropeller('data/apc_props/PER3_5x75E.dat')
bla = PropAPC2(prop.max_rpm, prop.max_vel, 16)
#bla.forward(torch.tensor([3000.0, 10.0]))

train(bla, prop, prop.max_rpm)
compare_thrust_curve(bla, prop)
