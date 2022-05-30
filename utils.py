import os
import sys
import json
import torch
from scipy.stats import truncnorm
from functools import wraps


def make_noise(batch, dim, truncation=None):
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim)
    else:
        return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)

      
def is_conditional(G):
    if hasattr(G, 'model'):
        return 'biggan' in G.model.__class__.__name__.lower()
    else:
        return 'biggan' in G.__class__.__name__.lower()


def one_hot(dims, value, indx):
    vec = torch.zeros(dims)
    vec[indx] = value
    return vec


def save_command_run_params(args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')


def truncated_noise(size, truncation=1.0):
    return truncnorm.rvs(-truncation, truncation, size=size)


def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method
