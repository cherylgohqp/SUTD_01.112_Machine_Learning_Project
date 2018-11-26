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
    :return: dic in the form: { (y_1y_2, yi): value }
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
    :param _model: from Model(file).train()
    :return: dataframe table
    '''

    perm_data = GenerateFile(_model)
    # print(perm_data)
    labels = ['__START__', 'O', 'B-positive', 'I-positive', '__STOP__',
              'B-negative', 'I-negative', 'B-neutral', 'I-neutral']

    data = [['']]
    # build top-most
    for y2y1y0, value in perm_data.items():
        data[0].append(y2y1y0[0])

    for label1 in labels:
        add_data = [label1]
        for y2y1y0, value in perm_data.items():
            if y2y1y0[1] == label1:
                add_data.append(value)
            else:
                add_data.append(0.00)

        data.append(add_data)

    # Create dataframe
    df = np.array(data, dtype=object)

    return pd.DataFrame(data=df[1:, 1:],
                        index=df[1:, 0],
                        columns=df[0, 1:])