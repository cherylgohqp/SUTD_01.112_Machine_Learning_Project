import numpy as np
import pandas as pd
# import networkx.drawing.nx_pydot as gl
# import networkx as nx
# import matplotlib.pyplot as plt
# from pprint import pprint
##matplotlib inline


# todo: finish placeholder
def Count(_training_set, _y, _x):
    # unsure what form this is
    if _y == _x:
        # count number of times this occurs in training set
        pass
    else:
        # count number of times in training set y is mapped to x
        pass


# part a
# returns e(x|y) = Count(x -> y)/Count(y)
def estimateEmissionParameters(_training_set, _x, _y):
    return Count(_training_set, _y, _x)/Count(_training_set, _y, _y)


# part b
# handle case whereby words appear in training set but not in test set
def generateUNK():
    pass


if __name__ == "__main__":
    pass
