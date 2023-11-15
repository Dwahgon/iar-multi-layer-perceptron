import sys
import matplotlib.pyplot as plt
import os.path
from args import args

def save_or_show(name):
    if args.out_dir:
        plt.savefig(os.path.join(args.out_dir, name))
    else:
        plt.show()