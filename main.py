from __future__ import print_function
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time

from nets_a import LinearNet1
import nets_b
import params

class T0_0(nn.Module): #2 linears
    def __init__(self):
        super(T0_0, self).__init__()
        self.fc0 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        return x
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:

            epochstats = [epoch, loss.item()]

            if args.dry_run:
                break
    return epochstats


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return {
        "testloss": test_loss,
        "accuracy": 100. * correct / len(test_loader.dataset)
    }


def main(**myargs):
    trainstats = []
    teststats = []
    t1 = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    t2 = time.time()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = LinearNet1(myargs).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    t3 = time.time()
    for epoch in range(1, args.epochs + 1):
        trainstats.append(train(args, model, device, train_loader, optimizer, epoch))
        teststats.append(test(model, device, test_loader))
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    t4 = time.time()

    stats = {
        "argparse": t2 - t1,
        "dataloading": t3 - t2,
        "modelling": t4 - t3,
        "total": t4 - t1,
        "trainstats": trainstats,
        "teststats": teststats
    }
    return stats


if __name__ == '__main__':

    t_zero = {"nothing": [],
              "relu": [],
              "softmax": [],
              "both": []}

    t_one = {"nothing": [],
             "relu": [],
             "softmax": [],
             "both": []}

    t_two = {"nothing": [],
             "relu": [],
             "softmax": [],
             "both": []}
    # for testing with nets_a.py
    # testing loops are set up to optimize the intuitiveness and ease of use of the results rather than code cleanliness

    # test zero -- COMPLETED
    print("Test Zero")
    for i in range(3):

        t_zero["nothing"].append(main(test_num=0))


    print("25%")
    for i in range(3):
        try:
            t_zero["relu"].append(main(use_relu=True, test_num=0))
        except Exception as e:
            print(e)
    print("50%")
    for i in range(3):
        try:
            t_zero["softmax"].append(main(use_softmax=True, test_num=0))
        except Exception as e:
            print(e)
    print("75%")
    for i in range(3):
        try:
            t_zero["both"].append(main(use_relu=True, use_softmax=True, test_num=0))
        except Exception as e:
            print(e)
    print("100%")

    with open("test_0.json", "w") as f:
        json.dump(t_zero, f, indent=4)


    # test one
    print("Test One")
    for z in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        for i in range(3):
            try:
                t_one["nothing"].append(main(test_num=1, two_layers_neuron_count=z))
            except Exception as e:
                print(e)
        print(f"25% done for z = {z}")
        for i in range(3):
            try:
                t_one["relu"].append(main(use_relu=True, test_num=1, two_layers_neuron_count=z))
            except Exception as e:
                print(e)
        print(f"50% done for z = {z}")
        for i in range(3):
            try:
                t_one["softmax"].append(main(use_softmax=True, test_num=1, two_layers_neuron_count=z))
            except Exception as e:
                print(e)
        print(f"75% done for z = {z}")
        for i in range(3):
            try:
                t_one["both"].append(
                    main(use_relu=True, use_softmax=True, test_num=1, two_layers_neuron_count=z))
            except Exception as e:
                print(e)
        print(f"100% done for z = {z}")

    with open("test_1.json", "w") as f:
        json.dump(t_one, f, indent=4)


    # test two
    print("Test Two")
    for i in range(3):
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                t_two["nothing"].append(main(test_num=2, intermediate_layer_count=j))
            except Exception as e:
                print(e)
    print("25%")
    for i in range(3):
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                t_two["relu"].append(main(use_relu=True, test_num=2, intermediate_layer_count=j))
            except Exception as e:
                print(e)
    print("50%")
    for i in range(3):
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                t_two["softmax"].append(main(use_softmax=True, test_num=2, intermediate_layer_count=j))
            except Exception as e:
                print(e)
    print("75%")
    for i in range(3):
        for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            try:
                t_two["both"].append(
                    main(use_relu=True, use_softmax=True, test_num=2, intermediate_layer_count=j))
            except Exception as e:
                print(e)
    print("100%. All Tests Completed.")


    allstats = {
        "test0": t_zero,
        "test1": t_one,
        "test2": t_two
    }
    with open("test_2.json", "w") as f:
        json.dump(allstats, f, indent=4)
