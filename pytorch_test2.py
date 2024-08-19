import torch

class table(torch.nn.Module):
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

        return torch.sum(mask * self.values)
    
def main():
    def fun(point):
        return torch.sin(point[0]) * torch.exp(torch.cos(point[1]))

    model = table(torch.linspace(0, 5, 10), torch.linspace(0, 5, 10))
    optimizer = torch.optim.Adam([model.values], lr=0.1)
    ave_loss = 0
    for epoch in range(1000):

        optimizer.zero_grad()

        point = torch.rand((2)) * 5
        
        loss = ( model(point) - fun(point) )**2

        loss.backward()

        optimizer.step()

        ave_loss += loss

        if (epoch + 1) % 100 == 0:
            print(ave_loss/100)
            ave_loss = 0

if __name__ == "__main__":
    main()       
