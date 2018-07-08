import torch
import torch.optim as optim
import argparse
from models.nets import NetVLAD_FC_GATE


#hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=32)
parser.add_argument('--test-batch-size', default=1000)
parser.add_argument('--epochs',default=10)
parser.add_argument('--lr',default=0.001)
parser.add_argument('--seed', default=1)
parser.add_argument('--log-interval', default=10)
args = parser.parse_args()

assert torch.cuda.is_available()
device = torch.device("cuda")

#dataloader
train_loader = ???
test_loader = ???

#model
model = NetVLAD_FC_GATE().to(device)
any_loss = ???

#optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

#trian
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = any_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#test
def test():
    pass


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
