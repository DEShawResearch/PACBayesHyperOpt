#!/usr/bin/env python
# python 3.7, pytorch 1.2, cuda 10.0, higher 0.5.1
# https://github.com/facebookresearch/higher

from __future__ import annotations
import numpy as np
import os
import argparse
import higher
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from typing import List, Tuple
import copy


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', default='MNIST', choices={"MNIST", "CIFAR-10"}, help="which dataset to fit")
    p.add_argument('--network', default='linear', choices={"linear", "resnet"}, help="which network architecture to use")
    p.add_argument('--inner_steps', default=1000, type=int)
    p.add_argument('--outer_steps', default=100, type=int)
    p.add_argument('--zeta', default=0, type=float, help="value of zeta scale factor to test")
    p.add_argument('--backprop_truncation', default=0, type=int, help="number of steps of backpropagation to truncate to (must divide inner_steps)")
    args = p.parse_args()

    backprop_trunc = args.backprop_truncation
    train_stats, valid_stats, test_stats, regularizers = run_fit(
        args.zeta,
        args.inner_steps,
        args.outer_steps,
        args.network,
        args.dataset,
        backprop_trunc=backprop_trunc + 1)

    with open(f'{zeta}_{backprop_trunc}.pkl', 'wb') as fp:
        pickle.dump((train_stats, valid_stats, test_stats, regularizers), fp)


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
        network (str) name of the network, either linear or resnet18
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
    else:
        raise ValueError()
    return model


def compute_regularizer_term(model: LinearNet, criterion: torch.nn.modules.loss.CrossEntropyLoss,
                             train_data: Tuple[torch.Tensor, torch.Tensor],
                             valid_data: Tuple[torch.Tensor, torch.Tensor],
                             weight_decays: List[torch.Tensor], zeta: float,
                             lr: float) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Computes the value of the regularizer in Eq. 7 and is called at each step of the
    inner optimization.

    Note: gradients here can be tracked using the Higher library

    Returns:
        norm (scalar-valued torch.tensor) this is the squared 2-norm of the difference
            between the training and validation gradients (Y in Algorithm 3.1).
    """
    train_images, train_labels = train_data
    valid_images, valid_labels = valid_data

    # compute training and validation gradient
    train_outputs = model(train_images)
    valid_outputs = model(valid_images)

    train_loss = criterion(train_outputs, train_labels)
    valid_loss = criterion(valid_outputs, valid_labels)

    train_grads = torch.autograd.grad(
        train_loss, model.parameters(), retain_graph=True, create_graph=True)
    valid_grads = torch.autograd.grad(
        valid_loss, model.parameters(), retain_graph=True, create_graph=True)

    norm = 0
    for p, wd, t_grad, v_grad in zip(model.parameters(), weight_decays, train_grads, valid_grads):
        a = (t_grad + wd * p.data) - v_grad
        norm += (a**2).sum()

    return norm


def optimizer_step(optimizer: torch.optim.Optimizer, model: LinearNet,
                   criterion: torch.nn.modules.loss.CrossEntropyLoss,
                   data: Tuple[torch.Tensor, torch.Tensor]):
    images, labels = data
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def differentiable_optimizer_step(
        optimizer: higher.optim.DifferentiableOptimizer,
        model: higher.patch.FunctionalLinearNet,
        criterion: torch.nn.modules.loss.CrossEntropyLoss,
        data: Tuple[torch.Tensor, torch.Tensor],
        weight_decays: List[torch.Tensor],
):
    images, labels = data
    outputs = model(images)

    regularizer = 0

    #    Uncomment me to run with coupled weight decay
    #    for wd, p in zip(weight_decays, model.parameters()):
    #        regularizer += 0.5 * (wd * (p**2)).sum()

    loss = criterion(outputs, labels)
    optimizer.step(loss + regularizer)

    # decoupled weight decay
    for p, wd in zip(model.parameters(), weight_decays):
        p.data -= 1e-4 * wd * p.data

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


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_fit(zeta: float,
            inner_steps: int,
            outer_steps: int,
            network : str,
            dataset : str,
            backprop_trunc: int = 1) -> Tuple[List, List, List, List]:
    """Run a single weight-decay experiment with a specific choice of the regularizer
    using Algorithm C.2 (offline optimization of Eq. 7). Uses the higher package to compute
    the derivative of the regularizer in Eq. 7 with truncated backpropagation.

    Args:
        zeta (float) the scale-factor that determines the strength of the regularization penalty
        inner_steps (int) number of steps of inner parameter optimization
        outer_steps (int) number of steps of the outer hyperparameter optimization to run
        network (str) either "linear" or "resnet"
        dataset (str) either "MNIST" or "CIFAR-10"
        backprop_trunc (int) the number of inner steps to backprop through for regularizer gradient estimation.

    Returns:
        train_stats, valid_stats, test_stats (Tuple[List, List]) each of these is a tuple
            containing the cross-entropy losses and classication accuracies over the inner
            optimization on the training, validation and test sets respectively.
        regularizers (List) value of the regularizer over the inner optimization
    """
    seed_everything(0)
    if dataset == 'MNIST':
        train_data, valid_data, testloader = fetch_MNIST_data(network)
    elif dataset == 'CIFAR-10':
        train_data, valid_data, testloader = fetch_CIFAR_data(network)

    valid_images, valid_labels = valid_data

    model = load_model(network, dataset)
    model = model.cuda()

    old_state_dict = copy.deepcopy(model.state_dict())

    criterion = torch.nn.CrossEntropyLoss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 1e-4 from Lorraine et al. (2019)

    weight_decays = []
    for parameter in model.parameters():
        weight_decays.append(torch.zeros_like(parameter, requires_grad=True))

    ho_optimizer = torch.optim.RMSprop(weight_decays, lr=1e-2)  # 1e-2 from Lorraine et al. (2019)

    ## penalty size
    zeta = zeta * lr * lr

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    test_losses = []
    test_accs = []
    regularizers = []

    inner_step_batch_size = backprop_trunc
    assert inner_steps % inner_step_batch_size == 0
    n_inner_batches = inner_steps // inner_step_batch_size

    for hyper_step in range(outer_steps):
        ho_optimizer.zero_grad()
        total_norm = 0
        regularizer_wd_grads = [torch.zeros_like(wd) for wd in weight_decays]

        for _ in range(n_inner_batches):
            with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
                batch_total_norm = 0
                for _b in range(inner_step_batch_size):
                    sq_norm = compute_regularizer_term(fmodel, criterion, train_data, valid_data,
                                                       weight_decays, zeta, lr)

                    batch_total_norm += sq_norm

                    # take training step with training gradient
                    train_loss = differentiable_optimizer_step(diffopt, fmodel, criterion,
                                                               train_data, weight_decays)

                # copy parameters from functional model back into main model
                # (without tracking for backprop)
                for p, fp in zip(model.parameters(), fmodel.parameters()):
                    p.data[:] = fp.data[:]

                # don't update optimizer state for the regularizer experiment (for now)
                for param_index, state_dict in diffopt.state[0].items():
                    optimizer.state[optimizer.param_groups[0]['params'][param_index]] = state_dict
                optimizer.zero_grad()

                # accumulating Y from Algorithm 3.1
                total_norm += batch_total_norm

                # accumulating X from Algorithm 3.1
                batch_weight_decay_grads = torch.autograd.grad(batch_total_norm, weight_decays)
                for i, g in enumerate(batch_weight_decay_grads):
                    regularizer_wd_grads[i] += zeta * g

        # take hyperparameter step
        valid_outputs = model(valid_images)
        valid_loss = criterion(valid_outputs, valid_labels)
        valid_loss.backward()

        # risk gradient estimate (comment me out to optimize only the regularizer)
        for wd, p in zip(weight_decays, model.parameters()):
            if wd.grad is None:
                wd.grad = torch.zeros_like(wd)
            wd.grad = -lr * p * p.grad

        # regularizer gradient estimate
        for wd, update in zip(weight_decays, regularizer_wd_grads):
            wd.grad += update / (2 * torch.sqrt(total_norm))

        ho_optimizer.step()

        # prep for data dump
        train_loss = train_loss.detach().cpu().numpy()
        valid_loss = valid_loss.detach().cpu().numpy()
        regularizer = torch.sqrt(total_norm).detach().cpu().numpy()

        train_acc, valid_acc, test_loss, test_acc = compute_and_log_statistics(
            model, train_data, valid_data, testloader, hyper_step, network, dataset)
        print(f'Hyper step: {hyper_step}')
        print(f'Valid Loss: {valid_loss}')
        print(f'Valid Accuracy: {valid_acc}')
        print(f'Train Accuracy: {train_acc}')
        print(f'Test Accuracy: {test_acc}')
        print(f'Regularizer: {torch.sqrt(total_norm)}')
        print(f'Objective: {valid_loss + zeta * regularizer}')
        print('-------------------------------')
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        regularizers.append(regularizer)

    train_stats = (train_losses, train_accs)
    valid_stats = (valid_losses, valid_accs)
    test_stats = (test_losses, test_accs)

    return train_stats, valid_stats, test_stats, regularizers

if __name__ == '__main__':
    main()
