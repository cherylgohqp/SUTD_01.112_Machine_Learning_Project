import numpy as np
import pandas as pd


# This generates based on 2nd order HMMs
def GenerateFile(_model):
    """
    Attempts to return a dict with 2nd order transition prob values,
    will put inside a tuple of form (prev1&prev2, current) as label
    Used:
    y_y1: { yi: { yi-1: count} }
    y_y2: { yi: {(yi-2, yi-1): count} }
    :param _model: Model used
    :return: dic in the form: { ((yi-2,yi-1), yi): value }
    """
    m = _model
    permutations = {}
    for yi, y2 in m.y_y2.items():  # from y_y2 as shown above
        for prev_labels, count in y2.items():
            yi_1 = prev_labels[1]
            yi_2 = prev_labels[0]
            value = count / (m.y_y1[yi_1][yi_2])  # P(yi | yi-2,yi-1) = Count(yi,yi-1,yi-2) / Count(yi-2, yi-1)
            permutations[(prev_labels, yi)] = value  # will save in the form { ((yi-2,yi-1),yi): value }

    return permutations


def GetTransitionDataFrame(_model):
    '''
    Part 4a - generating dataframe with all required transition params
    Generated result will look like:

    _ (yi-2, yi-1) ...
    y0
    y1
    yi
    ...
    ...

    :param _model: from Model(file).train()
    :return: dataframe table
    '''

    perm_data = GenerateFile(_model)
    # print(perm_data)
    labels = ['__START__', 'O', 'B-positive', 'I-positive', '__STOP__',
              'B-negative', 'I-negative', 'B-neutral', 'I-neutral']

    data = []
    # build top-most, need to filter out duplicates
    for y2y1y0, value in perm_data.items():
        if y2y1y0[0] not in data:
            data.append(y2y1y0[0])

    # create base df
    df = pd.DataFrame(data=0.00,
                      index=labels,
                      columns=data)

    # add all states/labels into their corresponding locations in df
    for label1 in labels:
        for d in data:
            try:
                df.loc[label1, [d]] = perm_data[(d, label1)]
            except KeyError:
                df.loc[label1, [d]] = 0.0

    return df
