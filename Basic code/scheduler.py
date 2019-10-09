from torch import optim
from torch import nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(5,3)

    def forward(self,x):
        return self.linear1(x)

model = Model()
optimizer = optim.Adam(model.parameters(), lr =1.0)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch : 0.95 ** epoch)

for epoch in range(1, 100+1):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    print('epoch: {:3d}, lr = {:.6f}'.format(epoch, lr))
    scheduler.step()