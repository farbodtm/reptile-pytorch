import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import data, io, img_as_ubyte

from miniimagenet import Model
import dataset as dset
import reptile

# Args TODO make them arguments.
seed = 0

log = False

DATA_DIR="./data/miniimagenet"

def argument_parser():
    """
    Parser for the script/
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test', help='evaluate a pre-trained model',
                        action='store_true', default=False)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--model-name', help='name of the model', default='normal')
    parser.add_argument('--classes', help='number of classes per inner task', default=5, type=int)
    parser.add_argument('--shots', help='number of examples per class', default=5, type=int)
    parser.add_argument('--train-shots', help='shots in a training batch', default=0, type=int)
    parser.add_argument('--inner-batch', help='inner batch size', default=5, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=20, type=int)
    parser.add_argument('--learning-rate', help='Adam step size', default=1e-3, type=float)
    parser.add_argument('--meta-step', help='meta-training step size', default=0.1, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=1, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=400000, type=int)
    parser.add_argument('--eval-batch', help='eval inner batch size', default=5, type=int)
    parser.add_argument('--eval-iters', help='eval inner iterations', default=50, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=10000, type=int)
    parser.add_argument('--eval-interval', help='train steps per eval', default=50, type=int)
    return parser

def train_kwargs(parsed_args):
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),

        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,

        'meta_step_size': parsed_args.meta_step,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,

        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
    }

def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,

        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,

        'num_samples': parsed_args.eval_samples,
    }

def main():
    """
    Load mini-imagenet and train a model.
    """

    # Parse arguments
    parser = argument_parser()
    args = parser.parse_args()
    t_kwargs = train_kwargs(args)
    e_kwargs = evaluate_kwargs(args)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    model_name = args.model_name
    net = Model(args.classes)
    if torch.cuda.is_available():
        net.cuda()
        #net = torch.nn.DataParallel(net).cuda()

    is_eval = args.test
    if is_eval:
       print "Evaluate mode"
       net.load_state_dict(torch.load("./models/model_r_{}.pth".format(model_name)))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0, 0.999))

    print net
    train_set, val_set, test_set = dset.read_dataset(DATA_DIR)
    print "Dataset loaded."

    if not is_eval:
        reptile.train(train_set, val_set, net, model_name, criterion, optimizer, **t_kwargs)
        torch.save(net.state_dict(), './models/model-{}.pth'.format(args.model_name))

    # Final eval
    print "accuracy on train_set: {}".format(reptile.test(train_set, net, criterion, optimizer, **e_kwargs))
    print "accuracy on val_set: {}".format(reptile.test(val_set, net, criterion, optimizer, **e_kwargs))
    print "accuracy on test_set: {}".format(reptile.test(test_set, net, criterion, optimizer, **e_kwargs))

if __name__ == '__main__':
    main()
