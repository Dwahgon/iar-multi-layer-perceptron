import argparse
import random
import os
import json

def dir_path(string):
    if os.path.isdir(string):
        return string
    raise NotADirectoryError(string)

def is_int_list(string):
    l = json.loads(string)
    if isinstance(l, list) and all(isinstance(x, int) for x in l):
        return l
    raise TypeError(string)

def is_float_list(string):
    l = json.loads(string)
    if isinstance(l, list) and all(isinstance(x, float) for x in l):
        return l
    raise TypeError(string)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-o', '--out_dir', help='Output directory of the graphs. If this argument is not passed, the graphs will be shown interactively', type=dir_path)
arg_parser.add_argument('-n', '--hidden_layers', help='Configure hidden layers. Syntax: [<hidden_layer_1_neurons>, <hidden_layer_2_neurons>, ..., <hidden_layer_n_neurons>].', type=is_int_list, default=[20, 20])
arg_parser.add_argument('-r', '--random_state', help='Random state used for reproducible results. If this argument is not passed, a random is generated', type=int, default=random.randint(0, 2**32 - 1))
arg_parser.add_argument('-t', '--test_size', help='Value between 0 and 1 representing the rate of the dataset that will be used as the test dataset', type=float, default=0.3)
arg_parser.add_argument('-l', '--learning_rates', help='List of learning rates that each model tested will use. Syntax: [<model_1_learning_rate>, <model_2_learning_rate>, ..., <model_n_learning_rate>]', type=is_float_list, default=[0.001, 0.01, 0.1, 0.3])

args = arg_parser.parse_args()