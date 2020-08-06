#!/usr/bin/env python
# python 3.7, pytorch 1.2, cuda 10.0
import numpy as np
import os
import argparse
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', default='MNIST', choices={"MNIST", "CIFAR-10"}, help="which dataset to fit")
    p.add_argument('--network', default='linear', choices={"linear", "resnet-18", "resnet-34"}, help="which network architecture to use")
    p.add_argument('--inner_steps', default=1000, type=int)
    p.add_argument('--outer_steps', default=100, type=int)
    p.add_argument('--online', default=False, action='store_true')
    p.add_argument('--zeta_min', default=0, type=float, help="minimum value of zeta to test")
    p.add_argument('--zeta_max', default=5, type=float, help="maximum value of zeta to test")
    p.add_argument('--num_zeta', default=1, type=int, help="number of values of zeta to test, linearly spaced between zeta_min and zeta_max")
    p.add_argument('--replicates', default=1, type=int, help="run each experiment this many times")
    args = p.parse_args()
    results_dict = {}
    for zeta in np.linspace(args.zeta_min, args.zeta_max, args.num_zeta):
        for fit_num in range(args.replicates):
            train_stats, valid_stats, test_stats, regularizers, w_norms = run_fit(
                zeta,
                args.inner_steps,
                args.outer_steps,
                args.online,
                args.network,
                args.dataset,
                fit_num=fit_num)
            result = {
                'train_stats': train_stats,
                'valid_stats': valid_stats,
                'test_stats': test_stats,
                'regularizers': regularizers,
                'weight_norms': w_norms
            }
            results_dict[(zeta, fit_num)] = result

    with open(f'results.pkl', 'wb') as fp:
        pickle.dump(results_dict, fp)


class LinearNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearNet, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, num_classes)  # 28*28 from image dimension, 10 classes

    def forward(self, x):
        x = self.fc1(x)
        return x


def get_accuracy(model, images, labels):
    """Compute the classification accuracy"""
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


def get_test_stats(model, testloader, network, dataset):
    """Compute classification accuracy for the entire test set

    Args:
        model (torch.nn) the model
        testloader (iterator) iterator over the test data
        network (str) name of the network, either linear, resnet-18, or resnet-34
        dataset (str) either MNIST or CIFAR-10

    Returns:
        test_loss (float) cross-entropy loss
        test_accuracy (float) fraction of test images classified correctly
    """
    dataloader = iter(testloader)
    total = 0
    correct = 0
    loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    for test_images, test_labels in dataloader:
        if network == 'linear':
            test_images = reshape_for_linear(test_images).cuda()
        elif dataset == 'MNIST':
            test_images = test_images.view(-1, 1, 28, 28).cuda()
        else:
            test_images = test_images.cuda()

        test_labels = test_labels.cuda()
        outputs = model(test_images)
        _, predicted = torch.max(outputs, 1)
        loss += criterion(outputs, test_labels)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()
    return loss, correct / total


def reshape_for_linear(images):
    """Reshape the images for the linear model
    Our linear model requires that the images be reshaped as a 1D tensor
    """
    n_images, n_rgb, img_height, img_width = images.shape
    return images.reshape(n_images, n_rgb * img_height * img_width)


def fetch_CIFAR_data(network='linear'):
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    dataiter = iter(trainloader)
    train_images, train_labels = dataiter.next()
    valid_images, valid_labels = dataiter.next()

    if network == 'linear':
        train_images = reshape_for_linear(train_images)
        valid_images = reshape_for_linear(valid_images)

    train_images = train_images.cuda()
    valid_images = valid_images.cuda()
    train_labels = train_labels.cuda()
    valid_labels = valid_labels.cuda()
    return (train_images, train_labels), (valid_images, valid_labels), testloader


def fetch_MNIST_data(network='linear'):
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=2)
    dataiter = iter(trainloader)
    train_images, train_labels = dataiter.next()
    valid_images, valid_labels = dataiter.next()

    if network == 'linear':
        train_images = reshape_for_linear(train_images).cuda()
        valid_images = reshape_for_linear(valid_images).cuda()
    else:
        train_images = train_images.view(-1, 1, 28, 28).cuda()
        valid_images = valid_images.view(-1, 1, 28, 28).cuda()

    train_labels = train_labels.cuda()
    valid_labels = valid_labels.cuda()
    return (train_images, train_labels), (valid_images, valid_labels), testloader


def compute_auxiliary(model, criterion: torch.nn.modules.loss.CrossEntropyLoss,
                      train_data: Tuple[torch.Tensor, torch.Tensor],
                      valid_data: Tuple[torch.Tensor, torch.Tensor],
                      weight_decays: List[torch.Tensor], zeta: float,
                      lr: float) -> torch.Tensor:
    """Compute the auxiliary variables described in Algorithm 3.1 in the main paper

    Note: this increments the gradient of each of the weight_decays tensors corresponding
    to the variable denoted as X in Algorithm 3.1

    Returns:
        norm (scalar-valued torch.tensor) this is the squared 2-norm of the difference
            between the training and validation gradients (Y in Algorithm 3.1).
    """
    train_images, train_labels = train_data
    valid_images, valid_labels = valid_data

    # compute training gradient
    model.zero_grad()
    outputs = model(train_images)
    loss = criterion(outputs, train_labels)
    loss.backward()

    train_grads = [p.grad.detach().clone() for p in model.parameters()]

    # compute validation gradient
    model.zero_grad()
    outputs = model(valid_images)
    loss = criterion(outputs, valid_labels)
    loss.backward()
    valid_grads = [p.grad for p in model.parameters()]
    """
    This code computes auxiliary variables for an update to the hyperparameter gradient.
    Tried to write the update rule here but it doesn't really work well in non-LaTeX.
    """
    align = []
    norm = 0
    for p, wd, t_grad, v_grad in zip(model.parameters(), weight_decays, train_grads, valid_grads):
        a = t_grad + wd * p.data - v_grad
        norm += (a**2).sum()
        align.append(a)

    for a, p, wd in zip(align, model.parameters(), weight_decays):
        if wd.grad is None:
            wd.grad = torch.zeros_like(wd)
        if norm == 0:
            wd.grad += 0
        else:
            wd.grad += zeta * p * a
    return norm


def optimizer_step(optimizer: torch.optim.Optimizer, model,
                   criterion: torch.nn.modules.loss.CrossEntropyLoss,
                   data: Tuple[torch.Tensor, torch.Tensor]):
    """Helper function to run a single step of the inner optimization"""
    images, labels = data
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def compute_and_log_statistics(model, train_data: Tuple[torch.Tensor, torch.Tensor],
                               valid_data: Tuple[torch.Tensor, torch.Tensor], testloader,
                               hyper_step: int, network: str, dataset: str):
    train_images, train_labels = train_data
    valid_images, valid_labels = valid_data
    with torch.no_grad():
        train_acc = get_accuracy(model, train_images, train_labels)
        valid_acc = get_accuracy(model, valid_images, valid_labels)

        test_loss, test_acc = get_test_stats(model, testloader, network, dataset)
        test_loss = test_loss.detach().cpu().numpy()
    return train_acc, valid_acc, test_loss, test_acc


def load_model(network, dataset):
    from torchvision import models
    if network == 'linear':
        if dataset == 'MNIST':
            model = LinearNet(28 * 28, 10)
        elif dataset == 'CIFAR-10':
            model = LinearNet(3 * 32 * 32, 10)

    elif network == 'resnet-18':
        model = models.resnet18()
        if dataset == 'MNIST':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif network == 'resnet-34':
        model = models.resnet34()
        if dataset == 'MNIST':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        raise ValueError("Invalid choice of network")
    return model


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_fit(zeta: float,
            inner_steps: int,
            outer_steps: int,
            online: bool = False,
            network: str = 'linear',
            dataset: str = 'MNIST',
            fit_num: int = 0) -> Tuple[List, List, List, List]:
    """Run a single weight-decay experiment with a specific choice of the regularizer
    using Algorithm C.3 (online optimization of Eq. 7)

    Args:
        zeta (float) the scale-factor that determines the strength of the regularization penalty
        inner_steps (int) number of steps of inner parameter optimization
        outer_steps (int) number of steps of the outer hyperparameter optimization to ru n
        network (str) either "linear" or "resnet-18" or "resnet-34"
        dataset (str) either "MNIST" or "CIFAR-10"
        fit_num (int) a random seed for parameter initialization.

    Returns:
        train_stats, valid_stats, test_stats (Tuple[List, List]) each of these is a tuple
            containing the cross-entropy losses and classication accuracies over the inner
            optimization on the training, validation and test sets respectively.
        regularizers (List) value of the regularizer over the inner optimization
    """
    # make sure we get a fixed random seed when loading the data, so that each fit gets the same
    # 50 image subset of MNIST or CIFAR10
    seed_everything(0)
    if dataset == 'MNIST':
        train_data, valid_data, testloader = fetch_MNIST_data(network)
    elif dataset == 'CIFAR-10':
        train_data, valid_data, testloader = fetch_CIFAR_data(network)

    valid_images, valid_labels = valid_data

    # Now set the random seed to something different so that each experiment
    # is independent for computing error bars over multiple parameter initializations
    seed_everything(fit_num)

    model = load_model(network, dataset)
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weight_decays = []
    for parameter in model.parameters():
        weight_decays.append(torch.zeros_like(parameter, requires_grad=True))

    ho_optimizer = torch.optim.RMSprop(weight_decays, lr=1e-2)

    ## penalty size
    zeta = zeta * lr**2

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    test_losses = []
    test_accs = []
    regularizers = []
    weight_norms = []

    for hyper_step in range(outer_steps):
        ho_optimizer.zero_grad()
        total_norm = 0
        for _ in range(inner_steps):

            sq_norm = compute_auxiliary(model, criterion, train_data, valid_data, weight_decays,
                                        zeta, lr)
            sq_norm = sq_norm.detach()
            if zeta > 0 and online:
                for wd in weight_decays:
                    wd.grad /= torch.sqrt(sq_norm)
                ho_optimizer.step()
                ho_optimizer.zero_grad()
            total_norm += sq_norm # Y update in Algorithm 3.1

            # take training step with training gradient
            train_loss = optimizer_step(optimizer, model, criterion, train_data)

            # add separable weight decay to optimizer step
            for p, wd in zip(model.parameters(), weight_decays):
                p.data -= lr * wd * p.data

        # take hyperparameter step
        optimizer.zero_grad()
        outputs = model(valid_images)
        valid_loss = criterion(outputs, valid_labels)
        valid_loss.backward()

        for wd in weight_decays:
            if zeta > 0 and not online:
                wd.grad /= 2 * torch.sqrt(total_norm)

        # validation risk hyperparameter gradient using identity approximation to the Hessian
        # gradient estimator proposed by Lorraine et al. 2019
        weight_norm = 0.0
        for wd, p in zip(weight_decays, model.parameters()):
            wd.grad += -lr * p * p.grad
            weight_norm += torch.norm(p)**2
        ho_optimizer.step()

        # prep for data dump
        train_loss = train_loss.detach().cpu().numpy()
        valid_loss = valid_loss.detach().cpu().numpy()
        regularizer = torch.sqrt(total_norm).detach().cpu().numpy()
        weight_norm = torch.sqrt(weight_norm).detach().cpu().numpy()

        train_acc, valid_acc, test_loss, test_acc = compute_and_log_statistics(
            model, train_data, valid_data, testloader, hyper_step, network, dataset)
        print(f'Hyper step: {hyper_step}')
        print(f'Valid Loss: {valid_loss}')
        print(f'Valid Accuracy: {valid_acc}')
        print(f'Train Accuracy: {train_acc}')
        print(f'Test Accuracy: {test_acc}')
        print(f'Regularizer: {torch.sqrt(total_norm)}')
        print('-------------------------------')
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        regularizers.append(regularizer)
        weight_norms.append(weight_norm)

    train_stats = (train_losses, train_accs)
    valid_stats = (valid_losses, valid_accs)
    test_stats = (test_losses, test_accs)

    return train_stats, valid_stats, test_stats, regularizers, weight_norms


if __name__ == '__main__':
    main()
