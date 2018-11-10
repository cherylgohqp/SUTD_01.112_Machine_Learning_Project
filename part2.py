import numpy as np
import pandas as pd
import document
# import networkx.drawing.nx_pydot as gl
# import networkx as nx
# import matplotlib.pyplot as plt
# from pprint import pprint
##matplotlib inline


# The following is a rough draft of part 2

# part b
# handle case whereby words appear in training set but not in test set
def UNKHandler(_d, _test, _y, _x):
    d = document.Document(_file=_test, _document=_d)
    k = d.parseWithTrainedData()
    try:
        y_count = _d.y[_y]
        y_x_count = _d.x[_x][_y]
        return y_count, y_x_count
    except:
        # unrecognised word
        y_count = _d.y[_y]
        return k/(y_count + k)


def createDocument(_data):
    d = document.Document(_data)
    d.parse()
    return d


def Count(d, _y, _x):
    try:
        y_count = d.y[_y]
        y_x_count = d.x[_x][_y]
        return y_count, y_x_count
    except:
        print("Exception")
        print(d.y)
        return 0, 1


# part a
# returns e(x|y) = Count(x -> y)/Count(y)
def emissionParameters(_document, _y, _x, _test=None):
    try:
        y_count = _document.y[_y]
        y_x_count = _document.x[_x][_y]
        return y_x_count / y_count
    except:
        print("Exception")
        print(_document.y)
        return UNKHandler(_d=_document, _y=_y, _x=_x, _test=_test)


def train(_data):
    # trains using training data
    d = createDocument(_data)
    d.parse()
    return d


if __name__ == "__main__":
    d = train("SG/train")
    d.document = d
    d.file = "EN/train"
    k = d.parseWithTrainedData()
    print(emissionParameters(d, 'O', 'food'))
