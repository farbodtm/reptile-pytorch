from copy import deepcopy

import torch
import dataset as dset
import numpy as np

def to_tensor(x, l=False):
    """
    Convert a numpy array to torch tensor.
    """
    t = torch.Tensor(x)
    if l:
        t = torch.LongTensor(x)
    if torch.cuda.is_available():
        return torch.autograd.Variable(t).cuda()
    return torch.autograd.Variable(t)

def inner_train_step(model, criterion, optimizer, batch):
    """
    Inner training step procedure.
    """

    x, y = zip(*batch)
    x = np.array(x)
    y = np.array(y, dtype=np.int64)

    x = to_tensor(x)
    y = to_tensor(y, True)

    ypred = model(x)
    loss = criterion(ypred, y) 
    #if log: print loss.data[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def predict(model, x, labels=None, criterion=None):
    """
    Predict class for the image. returns loss optionally.
    """
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
    """
    Meta training step procedure.
    """

    weights_original = deepcopy(model.state_dict())
    new_weights = []
    for _ in range(meta_batch_size):
        mini_dataset = dset.sample_mini_dataset(train_set, num_classes, num_shots)
        for batch in dset.mini_batches(mini_dataset, inner_batch_size, inner_iters, False):
            inner_train_step(model, criterion, optimizer, batch)
        #if log: print 'Done'
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

def train(train_set,
        val_set,
        model,
        model_name,
        criterion,
        optimizer,
        num_shots,
        train_shots,
        num_classes,
        inner_batch_size,
        inner_iters,
        meta_iters,
        meta_step_size,
        meta_batch_size,
        eval_interval,
        eval_inner_iters,
        eval_inner_batch_size):
    """
    Meta training.
    """

    best = 0.
    # Reptile training loop
    for i in range(meta_iters):
        frac_done = float(i) / meta_iters
        current_step_size = meta_step_size * (1. - frac_done)

        meta_train_step(train_set, model, criterion, optimizer, (train_shots or num_shots), num_classes, inner_batch_size, inner_iters, current_step_size, meta_batch_size)

        if i % 50 == 0:
            print i+1

        # Periodically evaluate
        if i % (eval_interval*10) == 0:
            val_acc = test(val_set, model, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters, 200)
            print "accuracy on val_set: {}".format(val_acc)
            torch.save(model.state_dict(), './models/model_r_{}.pth'.format(model_name))
            if val_acc >= best:
                torch.save(model.state_dict(), './models/model_best_{}_{}.pth'.format(i, model_name))
                best = val_acc
        # Periodically evaluate
        if i % eval_interval == 0:
            accs = []
            losses = []
            for dataset in [train_set, val_set]:
                ncorrect, loss = evaluate(dataset, model, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters)
                accs.append(ncorrect/float(num_classes))
                losses.append(loss)

            print "-----------------------------"
            print "iteration               {}, step_size: {}".format(i+1, current_step_size)
            print "accuracy train: {} test: {}".format(accs[0], accs[1])
            print "loss train: {} test: {}".format(losses[0], losses[1])

def evaluate(dataset,
        model,
        criterion,
        optimizer,
        num_shots,
        num_classes,
        inner_batch_size,
        inner_iters):
    """
    Evaluation. Trains on eval training set and evaluates on a small number of test images.
    """

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
        eval_inner_batch_size,
        eval_inner_iters,
        num_samples):
    """
    Runs evaluation multiple times and returns the number of correct predictions.
    """

    total = 0.

    for _ in range(num_samples):
      ncorrect, loss = evaluate(dataset, model, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters)
      total += ncorrect
    return total / (num_samples * num_classes)

def predict(model, x, labels=None, criterion=None):
    """
    Predict class for the image. returns loss optionally.
    """
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

def evaluate_embedding(dataset,
        model,
        criterion,
        optimizer,
        num_shots,
        num_classes,
        inner_batch_size,
        inner_iters):
    """
    Evaluation. Trains on eval training set and evaluates on a small number of test images.
    """

    weights_original = deepcopy(model.state_dict())

    train_sets = []
    test_sets = []

    for _ in range(4):
        train_set, test_set = dset.split_train_test(
            dset.sample_mini_dataset(dataset, num_classes, num_shots+1))
        train_sets.append(train_set)
        test_sets.append(test_set)

    for train_set in train_sets:
        for batch in dset.mini_batches(train_set, inner_batch_size, inner_iters/4, False):
            inner_train_step(model, criterion, optimizer, batch)

    num_correct = 0.
    for train_set, test_set in zip(train_sets, test_sets):
        inputs_test, labels_test = zip(*test_set)
        inputs_train, labels_train = zip(*train_set)

        xt = to_tensor(np.array(inputs_train))
        embt = model.embedding(xt).data.cpu().numpy()
        embtg = embt.reshape(-1, embt.shape[-2]/num_classes, embt.shape[-1])
        avgs = np.mean(embtg, axis=1)

        xte = to_tensor(np.array(inputs_test))
        embte = model.embedding(xte).data.cpu().numpy()

        pred = []
        for emb in embte:
            d = np.linalg.norm(np.repeat(np.array([emb]), avgs.shape[0], axis=0) - avgs, axis=1)
            pred.append(np.argmin(d))
        preds = np.array(pred)

        num_correct += sum([float(pred == sample[1]) for pred, sample in zip(preds, test_set)])

    model.load_state_dict({ name: weights_original[name] for name in weights_original })
    return num_correct

def test_embedding(dataset,
        model,
        criterion,
        optimizer,
        num_shots,
        num_classes,
        eval_inner_batch_size,
        eval_inner_iters,
        num_samples):
    """
    Runs evaluation multiple times and returns the number of correct predictions.
    """

    total = 0.

    for _ in range(num_samples):
      ncorrect = evaluate_embedding(dataset, model, criterion, optimizer, num_shots, num_classes, eval_inner_batch_size, eval_inner_iters)
      total += ncorrect
    return total / (num_samples * num_classes * 4)
