import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

from model import Model
import dataset as dset

# Args TODO make them arguments.
seed = 0

plot = True

num_classes = 5
num_shots = 5
train_shots = 15

meta_batch_size = 5
meta_iters = 20000
meta_step_size = 0.1

inner_batch_size = 10
inner_iters = 8
learning_rate = 0.00022

eval_inner_batch_size = 15
eval_inner_iters = 88
eval_interval = 100

DATA_DIR="./data/miniimagenet"


def to_tensor(x):
    return ag.Variable(torch.Tensor(x))

def inner_train_step(model, batch):
    x, y = zip(*batch)
    x = to_tensor(x)
    y = to_tensor(y)

    model.zero_grad()
    ypred = model(x)
    loss = model.criterion(ypred, y) 
    loss.backward()
    model.optimizer.step()

def predict(x, labels=None):
    x = totorch(x)
    out = model(x)
    pred = torch.max(out, 1)[1].data.squeeze()
    loss = None
    if labels:
        loss = model.criterion(out, labels) 
    return pred, loss

def meta_train_step(train_set,
        model,
        num_shots,
        num_classes,
        inner_batch_size,
        inner_iters,
        meta_step_size,
        meta_batch_size):
    weights_original = deepcopy(model.state_dict())
    new_weights = []
    for _ in range(meta_batch_size):
        mini_dataset = dset.sample_mini_dataset(train_set, num_classes, num_shots)
        weights_before = deepcopy(model.state_dict())
        for batch in dset.mini_batches(mini_dataset, inner_batch_size, inner_iters, False):
            inner_train_step(model, batch)
        weights.append(deepcopy(model.state_dict()))

    model.load_state_dict({ name: weights_before[name] for name in weights_before })
    for weights in new_weights:
        cur_weights = model.state_dict()
        model.load_state_dict({name : 
            cur_weights[name] + (weights[name] - cur_weights[name]) * meta_batch_size for name in cur_weights})


def evaluate(dataset,
        model,
        num_shots,
        num_classes,
        inner_batch_size,
        inner_iters):
    weights_original = deepcopy(model.state_dict())
    train_set, test_set = dset.split_train_test(
            dset.sample_mini_dataset(dset, num_classes, num_shots+1))

    weights_before = deepcopy(model.state_dict())
    for batch in dset.mini_batches(train_set, inner_batch_size, inner_iters, False):
        inner_train_step(model, batch)
    
    inputs, labels = zip(*test_set)
    preds, loss = predict(inputs, labels)
    num_correct = sum([(pred == sample[1] for pred, sample in zip(preds, test_set))])
    acc = num_correct / num_classes

    model.load_state_dict({ name: weights_before[name] for name in weights_before })
    return num_correct, loss


def main():
    """
    Load mini-imagenet and train a model.
    """

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model = Model(num_classes, learning_rate)

    train_set, val_set, test_set = dset.read_dataset(DATA_DIR)

    # Reptile training loop
    for i in range(meta_iters):
        frac_done = i / meta_iters
        current_step_size = meta_step_size * (1 - frac_done)
        meta_train_step(train_set, model, num_shots, num_classes, inner_batch_size, inner_iters, current_step_size, meta_batch_size)

        # Periodically evaluate
        if i % eval_interval == 0:
            accs = []
            losses = []
            for dataset in [train_set, test_set]:
                acc, loss = evaluate(t, model, num_shots, num_classes, inner_batch_size, inner_iters)
                accs.append(ncorrect / num_classes)
                losses.append(loss)

            print(f"-----------------------------")
            print(f"iteration               {i+1}")
            print(f"accuracy train {accs[0]:.3f} test: {accs[1]:.3f}")
            print(f"loss train:{losses[0]:.3f} test: {losses[1]:.3f}")
    torch.save({'state_dict': model.state_dict(), 'optimizer': model.optimizer.state_dict()}, 'model.pth')

if __name__ == '__main__':
    main()
