# import the required packages
import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
import numpy as np


# define our data generation function
def data_generator(data_size=50):
    # f(x) = y = 8x^2 + 4x - 3    inputs = []
    labels = []

    # loop data_size times to generate the data
    for ix in range(data_size):
        # generate a random number between 0 and 1000
        x = np.random.randint(1000) / 1000

        # calculate the y value using the function 8x^2 + 4x - 3
        y = 8 * (x * x) + (4 * x) - 3

        # append the values to our input and labels lists
        inputs.append([x])
        labels.append([y])

    return inputs, labels


# define the model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = Linear(1, 6)
        self.fc2 = Linear(6, 6)
        self.fc3 = Linear(6, 1)

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()  # define the loss function
critereon = MSELoss()  # define the optimizer
optimizer = SGD(model.parameters(), lr=0.01)
# define the number of epochs and the data set size
nb_epochs = 200
data_size = 1000  # create our training loop
for epoch in range(nb_epochs):
    X, y = data_generator(data_size)

    epoch_loss = 0;

    for ix in range(data_size):
        y_pred = model(Variable(Tensor(X[ix])))

        loss = critereon(y_pred, Variable(Tensor(y[ix]), requires_grad=False))

        epoch_loss = loss.data[0]

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print("Epoch: {} Loss: {}".format(epoch, epoch_loss))  # test the model
model.eval()
test_data = data_generator(1)
prediction = model(Variable(Tensor(test_data[0][0])))
print("Prediction: {}".format(prediction.data[0]))
print("Expected: {}".format(test_data[1][0]))