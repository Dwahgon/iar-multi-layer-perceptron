import sys
import matplotlib.pyplot as plt
import os.path

args = sys.argv

def save_or_show(name):
    if "-o" in args:
        plt.savefig(os.path.join(args[args.index("-o") + 1], name))
    else:
        plt.show()