import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml

DEFAULT_ARGS = {
    'outputpath': 'experiments',
    'batch_size': 4,
    'num_workers': 6, # Number of dataset loaders
    'mixer_args': {},
    'n_saved': 1,
    'epochs': 400, 
    'optimizer': 'Adam',
    'optimizer_args': {
        'lr': 0.001,
    },
}


class half_lr_rate(object):
    def __init__(self, optimizer, param_name, score_function, patience = 3, min_delta = 0.0):
        self.optimizer = optimizer
        self.param_name = param_name
        self.score_function = score_function
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.min_delta = min_delta

    def __call__(self, evaluator):
        score = self.score_function(evaluator)

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group[self.param_name] = param_group[self.param_name] / 2.0
        else:
            self.best_score = score
            self.counter = 0


def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    arguments = dict(yaml_config, **kwargs)
    # In case some arguments were not passed, replace with default ones
    for key, value in DEFAULT_ARGS.items():
        arguments.setdefault(key, value)
    return arguments


