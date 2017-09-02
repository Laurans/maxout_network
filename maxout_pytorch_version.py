import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

# ------------- DATA ----------------
x_train = torch.from_numpy(np.expand_dims(np.linspace(-30,30),1)).float()
y_train = x_train**2

# Define constants of our maxout units
d_in = 1
d_out = 1
pool_size = 3

# -------------- GRAPH --------------
class Maxout(nn.Module):
    def __init__(self, d, m, k):
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d, m, k
        self.lin = nn.Linear(d, m * k)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(dim=max_dim)
        return m

model = Maxout(d_in, d_out, pool_size).cuda()
# ------------- LEARNING CONTEXT -------------
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# ------------- TRAINING SESSION -------------
for epoch in range(20000):
    running_loss = 0.0
    for instance, label in zip(x_train, y_train):
        var_x = autograd.Variable(instance.cuda())
        targets = autograd.Variable(label.cuda())
        model.zero_grad()
        outputs = model(var_x.resize(1, 1))
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
    if epoch % 1000 == 0:
        print(epoch, running_loss / x_train.size(0))

# ------------- VISUALIZING APPROXIMATION ----------
def graph(formula, linear, x_range):
    x = np.array(x_range)
    x = np.expand_dims(x, 1)
    y = formula(x, linear)
    for i in range(y.shape[-1]):
        plt.plot(x, y[:, i])
    plt.plot(x, quadratique(x))
    plt.show()

def quadratique(x):
    return x ** 2

def maxout_linears(x, l):
    W = l.weight.cpu().data.numpy()
    b = l.bias.cpu().data.numpy()
    return x * W.T + b

graph(maxout_linears, model.lin, range(-30,31))