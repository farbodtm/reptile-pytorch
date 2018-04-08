import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy

from skimage import data, io, img_as_ubyte
from matplotlib import pyplot as plt

from miniimagenet import Model
import dataset as dset

# Args TODO make them arguments.
seed = 0

cuda_set = True

num_classes = 5
num_shots = 5
train_shots = 15

meta_batch_size = 5
meta_iters = 200000
#meta_step_size = 0.1
meta_step_size = 1

inner_batch_size = 10
inner_iters = 8
learning_rate = 0.00022

eval_inner_batch_size = 15
eval_inner_iters = 88
eval_interval = 50

DATA_DIR="./data/miniimagenet"

log = False


def to_tensor(x, l=False):
    t = torch.Tensor(x)
    if l:
        t = torch.LongTensor(x)
    if torch.cuda.is_available() and cuda_set:
        return ag.Variable(t).cuda()
    return ag.Variable(t)

def inner_train_step(model, criterion, optimizer, batch):
    x, y = zip(*batch)
    x = np.array(x)
    y = np.array(y, dtype=np.int64)

    x = to_tensor(x)
    y = to_tensor(y, True)

    ypred = model(x)
    loss = criterion(ypred, y) 
    if log: print loss.data[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def predict(model, x, labels=None, criterion=None):
    x = np.array(x)
    x = to_tensor(x)
    out = model(x)
    pred = torch.max(out, 1)[1].data.squeeze()
    loss = None
    if labels:
        labels = np.array(labels, dtype=np.int64)
        labels = to_tensor(labels, True)
        loss = criterion(out, labels) 
        loss = loss.data[0]
    return pred, loss

def meta_train_step(train_set,
        model,
        criterion,
        optimizer,
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
        for batch in dset.mini_batches(mini_dataset, inner_batch_size, inner_iters, False):
            inner_train_step(model, criterion, optimizer, batch)
        if log: print 'Done'
        new_weights.append(deepcopy(model.state_dict()))
        model.load_state_dict({ name: weights_original[name] for name in weights_original })

    ws = len(new_weights)
    fweights = { name : new_weights[0][name]/float(ws) for name in new_weights[0] }
    for i in range(1, ws):
        #cur_weights = deepcopy(model.state_dict())
        for name in new_weights[i]:
            fweights[name] += new_weights[i][name]/float(ws)

    model.load_state_dict({name : 
        weights_original[name] + ((fweights[name] - weights_original[name]) * meta_step_size) for name in weights_original})


def evaluate(dataset,
        model,
        criterion,
        optimizer,
        num_shots,
        num_classes,
        inner_batch_size,
        inner_iters):
    weights_original = deepcopy(model.state_dict())
    train_set, test_set = dset.split_train_test(
            dset.sample_mini_dataset(dataset, num_classes, num_shots+1))

    for batch in dset.mini_batches(train_set, inner_batch_size, inner_iters, False):
        inner_train_step(model, criterion, optimizer, batch)
    
    inputs, labels = zip(*test_set)
    preds, loss = predict(model, inputs, labels, criterion)
    preds = preds.cpu().numpy()

    num_correct = sum([float(pred == sample[1]) for pred, sample in zip(preds, test_set)])

    model.load_state_dict({ name: weights_original[name] for name in weights_original })
    return num_correct, loss

def test(dataset,
        model,
        criterion,
        optimizer,
        num_shots,
        num_classes,
        inner_batch_size,
        inner_iters, samples):
    total = 0.

    for _ in range(samples):
      ncorrect, loss = evaluate(dataset, model, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters)
      total += ncorrect
    return total / (samples*num_classes)

def main():
    """
    Load mini-imagenet and train a model.
    """

    model_name = 'normal_3'
    rng = np.random.RandomState(seed)
    #torch.manual_seed(seed)

    net = Model(num_classes, learning_rate)
    if torch.cuda.is_available() and cuda_set:
        net.cuda()
        #net = torch.nn.DataParallel(net).cuda()

    eval = True
    if eval:
       net.load_state_dict(torch.load("./model_r_{}.pth".format(model_name)))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0, 0.999))

    print net
    train_set, val_set, test_set = dset.read_dataset(DATA_DIR)
    print "Dataset loaded."
    num_samples = 1000

    if eval:
        num_samples = 1000
        print "accuracy on train_set: {}".format(test(train_set, net, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters, num_samples))
        print "accuracy on val_set: {}".format(test(val_set, net, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters, num_samples))
        print "accuracy on test_set: {}".format(test(test_set, net, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters, num_samples))
        return
        

    # Reptile training loop
    for i in range(meta_iters):
        #frac_done = i / meta_iters
        frac_done = float(i) / meta_iters
        current_step_size = meta_step_size * (1. - frac_done)

        meta_train_step(train_set, net, criterion, optimizer, train_shots, num_classes, inner_batch_size, inner_iters, current_step_size, meta_batch_size)

        if i % 50 == 0:
            print i+1

        # Periodically evaluate
        if i % (eval_interval*10) == 0:
            print "accuracy on test_set: {}".format(test(test_set, net, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters, 200))
            torch.save(net.state_dict(), 'model_r_{}.pth'.format(model_name))
        # Periodically evaluate
        if i % eval_interval == 0:
            accs = []
            losses = []
            for dataset in [train_set, test_set]:
                ncorrect, loss = evaluate(dataset, net, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters)
                accs.append(ncorrect/float(num_classes))
                losses.append(loss)

            print "-----------------------------"
            print "iteration               {}, step_size: {}".format(i+1, current_step_size)
            print "accuracy train: {} test: {}".format(accs[0], accs[1])
            print "loss train: {} test: {}".format(losses[0], losses[1])

    print "accuracy on train_set: {}".format(test(train_set, net, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters, num_samples))
    print "accuracy on test_set: {}".format(test(test_set, net, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters, num_samples))
    print "accuracy on val_set: {}".format(test(val_set, net, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters, num_samples))
    torch.save(net.state_dict(), 'model-{}.pth'.format(model_name))

if __name__ == '__main__':
    main()
