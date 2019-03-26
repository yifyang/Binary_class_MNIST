import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision import transforms

from trainer import do
from datasets import SpecialMNIST
from networks import SoftmaxNet

parser = argparse.ArgumentParser(description='Difference in MNIST')
parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                    help='input batch size for training (default: 64)')
parser.add_argument('--shuffle-dataset', type=bool, default=True, metavar='Shuf',
                    help='enable shuffling dataset (default: True)')
parser.add_argument('--split-data', type=float, default=0.1, metavar='SP',
                    help='how much for testing (default: 0.1)')
parser.add_argument('--epoch', type=int, default=100, metavar='EP',
                    help='number of epochs for training (default: 100)')
parser.add_argument('--random-seed', type=int, default=42, metavar='R',
                    help='seed for shuffling dataset (default: 42)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--log_interval', type=float, default=10, metavar='LI',
                    help='log_interval (default: 10)')
parser.add_argument('--mean', type=float, default=0.1307, metavar='MEAN',
                    help='mean for transforming (default: 0.1307)')
parser.add_argument('--std', type=float, default=0.3081, metavar='STD',
                    help='std for transforming (default: 0.3081)')
parser.add_argument('--cuda', type=bool, default=False, metavar='STD',
                    help='enable cuda or not (default: False)')

if __name__ == "__main__":
    args = parser.parse_args()
    mean = args.mean
    std = args.std
    batch_size = args.batch_size
    shuffle_dataset = args.shuffle_dataset
    split_data = args.split_data
    epoch = args.epoch
    random_seed = args.random_seed
    lr = args.lr
    cuda = args.cuda
    log_interval = args.log_interval

    train_dataset = MNIST('../data/MNIST', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((mean,), (std,))
                                 ]))
    test_dataset = MNIST('../data/MNIST', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))
                                ]))

    # Returns pairs of images and target same/different
    new_train_dataset = SpecialMNIST(train_dataset)
    new_test_dataset = SpecialMNIST(test_dataset)

    # Get 10% of the whole dataset
    train_indices = list(range(len(train_dataset)))
    test_indices = list(range(len(test_dataset)))
    np.random.seed(random_seed)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    train_split = int(np.floor(split_data * len(train_dataset)))
    test_split = int(np.floor(split_data * len(test_dataset)))
    train_sample_indices = train_indices[:train_split]
    test_sample_indices = test_indices[:test_split]

    train_sampler = SubsetRandomSampler(train_sample_indices)
    test_sampler = SubsetRandomSampler(test_sample_indices)

    # Return the dataloader for training and validation.
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(new_train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(new_test_dataset, batch_size=batch_size, sampler=test_sampler, **kwargs)

    model = SoftmaxNet()
    if cuda:
        model.cuda()

    loss_fn = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    do(train_loader, test_loader, model, loss_fn, optimizer, epoch, cuda, log_interval)