import json
import math
import pickle

import matplotlib.pyplot as plt
import numpy
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# this is a command line program which can be run with different options
import argparse

from torch.utils.data import SubsetRandomSampler
from sklearn.manifold import TSNE

#Default lr: 0.001
VALID_SIZE = 0.1
MODEL_PATH = './models/'

# from zmq import device
parser = argparse.ArgumentParser()
parser.add_argument("-b","--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer")
parser.add_argument("--lr", type=float, default=math.pow(10,-3), help="learning rate")
parser.add_argument("-m", "--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--epochs", type=int, default=1, help="training epochs")


class Data():
    def __init__(self, args):
        # transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])        
        self.train_set = torchvision.datasets.MNIST('../data', download=True, train=True, transform=transform)
        self.test_set = torchvision.datasets.MNIST('../data', download=True, train=False, transform=transform)
        # dataloaders
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        # Task: split train_set into train_set and val_set and create loaders
        indices = list(range(len(self.train_set)))
        np.random.shuffle(indices)
        split = int(np.floor(VALID_SIZE * len(self.train_set)))

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self.train_loader_split = torch.utils.data.DataLoader(self.train_set, batch_size=args.batch_size, sampler=train_sampler,
                                                   num_workers=0)
        self.valid_loader_split = torch.utils.data.DataLoader(self.train_set, batch_size=args.batch_size, sampler=valid_sampler,
                                                   num_workers=0)


# Task: # Define a neural network model using the higher level building blocks: torch.nn.Sequential, torch.nn.Linear , torch.nn.Conv2d, nn.ReLU.
# Start with a simple model for ease of debugging. Chose an optimizer, for example torch.optim.SGD. Create the training loop.
# The basic variant, together with a simple network loss and optimizer are already implemented in the template.

# Refine the net architecture, using convolutions, max-pooling and a fully connected layer in the end.
# Report you architecture by calling print(net). Note, this is a very small dataset, with small input images.
# Do not use a full-blown architecture such as VGG that we considered in the lecture.
# Invent a small convolutional architecture of your own or get inspiration from the famous LeNet5 model.
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Linear(864, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(-1, 6 * 12 * 12)
        x = self.layer2(x)
        return x

def new_network(args):
    # net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    net = MyNet()
    net.to(args.dev)
    return net


# loss function
# This criterion combines nn.LogSoftmax and nn.NLLLoss in one single class.
# loss = nn.CrossEntropyLoss(reduction='none')

# optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# def new_network(args):
#     net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
#     net.to(args.dev)
#     return net

def save_network(net:nn.Module, filename):
    torch.save({'model_state_dict': net.state_dict()}, filename)

def load_network(args, filename):
    state = torch.load(open(filename, "rb") , map_location=args.dev)
    net = new_network(args)
    net.load_state_dict(state['model_state_dict'])
    return net


def evaluate(net: nn.Module, loader):
    ## Task: perform evaluation of the network on the data given by the loader
    net.eval()
    loss = 0
    acc = 0
    return loss, acc

def change_lr(batches, initial_lr, end_lr):
    return (end_lr - initial_lr) / batches

def train(args):
    print(args)
    data = Data(args)

    # lets verify how the loader packs the data
    (input, target) = next(iter(data.train_loader_split))
    # expect to get [batch_size x 1 x 28 x 28]
    print('Input  type:', input.type())
    print('Input  size:', input.size())
    # xpect to get [batch_size]
    print('Labels size:', target.size())
    # see number of trainig data:
    n_train_data = len(data.train_loader_split)
    print('Train data size:', n_train_data)

    # network, expect input images 28* 28 and 10 classes
    net = new_network(args)

    # validation data
    (input_val, target_val) = next(iter(data.valid_loader_split))
    n_val_data = len(data.valid_loader_split)

    # loss function
    loss = nn.CrossEntropyLoss(reduction='none')

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    history = {'train_loss_batch': np.empty((args.epochs, data.train_loader_split.__len__())), 'train_acc_batch': np.empty((args.epochs, data.train_loader_split.__len__()))}
    history_val = {'val_acc': np.empty((args.epochs)), 'val_loss': np.empty((args.epochs))}
    history_ewa = {'ewa_acc': np.empty((args.epochs, data.train_loader_split.__len__())),
               'ewa_loss': np.empty((args.epochs, data.train_loader_split.__len__()))}

    for epoch in range(args.epochs):
        # will accumulate total loss over the dataset
        L = 0
        # we will accumulate total accuracy over the dataset
        ACC = 0
        EWA_L = 0
        EWA_ACC = 0
        # loop fetching a mini-batch of data at each iteration
        train_loss_batch = np.empty((0,2))
        train_acc_batch = np.empty((0,2))
        for i, (x, target) in enumerate(data.train_loader_split):
            x = x.to(dev)
            target = target.to(dev)
            # apply the network
            y = net.forward(x)
            # calculate mini-batch losses
            l = loss(y, target)
            qt = 1/(i+1)
            # predict
            pr_energies, pr_labels = torch.max(y.data, 1)
            acc_current = (pr_labels == target).sum().item() / len(target)
            # Save results into history
            history['train_acc_batch'][epoch][i] = acc_current
            history['train_loss_batch'][epoch][i] = l.sum().item()/x.shape[0]
            EWA_ACC = (1-qt)*EWA_ACC + qt * acc_current
            EWA_L = (1-qt)*EWA_L + qt * (l.sum().item()/x.shape[0])
            history_ewa['ewa_acc'][epoch][i] = EWA_ACC
            history_ewa['ewa_loss'][epoch][i] = EWA_L
            ACC += acc_current
            # append loss and learning rate in current batch
            train_loss_batch = numpy.append(train_loss_batch,np.array([[optimizer.param_groups[0]['lr'], l.sum().item()/x.shape[0]]]), axis=0)
            train_acc_batch = numpy.append(train_acc_batch,np.array([[optimizer.param_groups[0]['lr'], acc_current]]), axis=0)
            # accumulate the total loss as a regular float number (important to sop graph tracking)
            L += l.mean().item()
            # the gradient usually accumulates, need to clear explicitly
            optimizer.zero_grad()
            # compute the gradient from the mini-batch loss
            l.mean().backward()
            # make the optimization step
            optimizer.step()

            # Change lr after each batch
            # optimizer.param_groups[0]['lr'] += change_lr(data.train_loader_split.__len__(),args.lr, end_lr=0.1)

        # set lr to default value after epoch
        # optimizer.param_groups[0]['lr'] = args.lr
        print(f'Epoch: {epoch} TRAIN mean loss: {L / n_train_data}, mean acc: {ACC / n_train_data}')
        torch.save(net.state_dict(), MODEL_PATH + f'baseline_ep{epoch + 1}.pt')

        # validation
        with torch.no_grad():
            loss_val = 0
            acc_val = 0
            for i, (x, target) in enumerate(data.valid_loader_split):
                x = x.to(dev)
                target = target.to(dev)
                # apply the network
                y = net.forward(x)

                l = loss(y, target)
                pr_energies, pr_labels = torch.max(y.data, 1)
                acc_current = (pr_labels == target).sum().item() / len(target)

                loss_val += l.mean()
                acc_val += acc_current

            acc_val_mean = acc_val / n_val_data
            loss_val_mean = loss_val / n_val_data
            print(f'Epoch: {epoch} VAL mean loss: {loss_val_mean}, mean acc: {acc_val_mean}')
            history_val['val_acc'][epoch] = acc_val_mean
            history_val['val_loss'][epoch] = loss_val_mean


    # with open('history.pkl', 'wb') as f:
    #     pickle.dump(history, f)
    #
    # with open('history_val.pkl', 'wb') as f:
    #     pickle.dump(history_val, f)
    #
    # with open('history_ewa.pkl', 'wb') as f:
    #     pickle.dump(history_ewa, f)

    save_network(net, 'network.pt')


        # Plot range test
        # plt.title("MyNet")
        # plt.xlabel("Learning rate")
        # plt.ylabel("Mini-Batch loss")
        # plt.xscale('log')
        # plt.plot(train_loss_batch[:,0],train_loss_batch[:,1], color="blue")
        # plt.show()
        # plt.savefig('train_loss_batch.pdf')

        # plt.title("MyNet")
        # plt.xlabel("Learning rate")
        # plt.ylabel("Accuracy")
        # plt.xscale('log')
        # plt.plot(train_acc_batch[:, 0], train_acc_batch[:, 1], color="blue")
        # plt.savefig('train_acc_batch.pdf')

def test_net(args):
    print(args)
    data = Data(args)
    ccm = np.zeros((10,10))
    # loss function
    loss = nn.CrossEntropyLoss(reduction='none')
    (x, target) = next(iter(data.test_loader))
    with torch.no_grad():
        x = x.to(dev)
        target = target.to(dev)
        # apply the network
        y = net.forward(x)

        l = loss(y, target)
        pr_energies, pr_labels = torch.max(y.data, 1)
        acc = (pr_labels == target).sum().item() / len(target)
        for i in range(len(target)):
            if pr_labels[i] != target[i]:
                ccm[pr_labels[i].item()][target[i].item()]+= 1

    print(f'Test mean loss: {l.mean()}, Test mean acc: {acc}')
    print(ccm)

def classifier_rank_well(args):
    print(args)
    data = Data(args)
    (x, target) = next(iter(data.test_loader))
    confidences = []
    errors = []
    with torch.no_grad():
        correct_test = 0
        for i, (x, target) in enumerate(data.test_loader):
            x = x.to(dev)
            target = target.to(dev)

            # fw test data
            y = net.forward(x)

            pr_energies, pr_labels = torch.max(y.data, 1)
            pr_probs = [F.softmax(el, dim=0).numpy() for el in y]
            pr_probs = np.array(pr_probs)

            confidences.append(np.max(pr_probs, axis=1))
            errors.append(1 * (pr_labels != target).numpy())

            correct_test += (pr_labels == target).sum().item()


        errors = np.hstack(errors)
        confidences = np.hstack(confidences)

        arr1inds = (confidences).argsort()
        confidences = confidences[arr1inds[::-1]]
        errors = errors[arr1inds[::-1]]

        err_cumsum = np.cumsum(errors)

        # plt.plot(confidences, err_cumsum)
        # plt.xlabel('reject decision confidence threshold')
        # plt.ylabel('num of errors')
        # plt.savefig('rejected.pdf')

        # plt.plot(np.flip(confidences), list(range(1, len(confidences) + 1)))
        # plt.xlabel('rejected decision confidence threshold')
        # plt.ylabel('rejected points count')
        # plt.savefig('rejected_points_count.pdf')

        # points_accepted = np.array(list(range(1, len(confidences) + 1)))
        # rel_err = err_cumsum / points_accepted
        # plt.plot(confidences, rel_err)
        # plt.ylabel("relative error #errors/#points_accepted")
        # plt.xlabel("rejected decision confidence threshold")
        # plt.savefig('rejected_decisions_confidence.pdf')

def tsne_plot(args):
    data = Data(args)
    (x, target) = next(iter(data.test_loader))
    modules = list(net.children())[::-1]
    print(nn.Sequential(*modules))
    with torch.no_grad():
        x = net.layer1(x)
        x = x.view(-1, 6 * 12 * 12)
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(x)
        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
        for i, c, label in zip([0,1,2,3,4,5,6,7,8,9], colors, ['0', '1', '2', '3','4','5','6','7','8','9']):
            plt.scatter(X_2d[target == i, 0], X_2d[target == i, 1], c=c, label=label)
        plt.legend()
        for i, c, label in zip([0,1,2,3,4,5,6,7,8,9], colors, ['0', '1', '2', '3','4','5','6','7','8','9']):
            plt.scatter(X_2d[target == i, 0], X_2d[target == i, 1], c=c, label=label)
        plt.legend()
    plt.savefig('tsne')

if __name__ == "__main__":
    args = parser.parse_args()
    # query if we have GPU
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', dev)
    args.dev = dev
    args.epochs = 30
    # train(args)
    net = MyNet()
    net.load_state_dict(torch.load('models/baseline_ep30.pt'))
    # classifier_rank_well(args)
    tsne_plot(args)

