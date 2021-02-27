import torch
import torch.nn as nn
import torchnet as tnt
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.1
DROPOUT = 0.25
MOMENTUM = 0.9  # heavy ball momentum in gradient descent
DATA_DIR = '/matt/data'
CUDA = torch.cuda.is_available()

# Data loaders
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1000, shuffle=True, **kwargs)


# The LeNet architecture, with dropout and batch normalization
class View(nn.Module):
    def __init__(self, o):
        super(View, self).__init__()
        self.o = o
    def forward(self, x):
        return x.view(-1, self.o)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        def convbn(ci, co, ksz, psz, p):
            return nn.Sequential(
                nn.Conv2d(ci, co, ksz),
                nn.BatchNorm2d(co),
                nn.ReLU(True),
                nn.MaxPool2d(psz, stride=psz),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1, 20, 5, 3, DROPOUT),
            convbn(20, 50, 5, 2, DROPOUT),
            View(50 * 2 * 2),
            nn.Linear(50 * 2 * 2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(DROPOUT),
            nn.Linear(500, 10))

    def forward(self, x):
        return self.m(x)


# Initialize the model, the loss function and the optimizer
model = LeNet()
loss_function = nn.CrossEntropyLoss()
if CUDA:
    model.cuda()
    loss_function.cuda()

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


# Function to train the model on one epoch of data
def train(epoch):
    model.train()
    for batch_ix, (data, target) in enumerate(train_loader):
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_ix % 100 == 0 and batch_ix > 0:
            # import pdb; pdb.set_trace()
            # print('[Epoch %2d, batch %3d] training loss: %.4f' %
            #       (epoch, batch_ix, loss.data[0]))
            print('[Epoch %2d, batch %3d] training loss: %.4f' %
                  (epoch, batch_ix, loss.data.item()))


# Test the model on one epoch of validation data
def test():
    model.eval()
    test_loss = tnt.meter.AverageValueMeter()
    top1 = tnt.meter.ClassErrorMeter()
    for data, target in test_loader:
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss = loss_function(output, target)

        top1.add(output.data, target.data)
        # test_loss.add(loss.data[0])
        test_loss.add(loss.data.item())

    print('[Epoch %2d] Average test loss: %.3f, accuracy: %.2f%%\n'
          % (epoch, test_loss.value()[0], top1.value()[0]))



if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test()
