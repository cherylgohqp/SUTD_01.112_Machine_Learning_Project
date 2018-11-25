import numpy as np
import pandas as pd


# This generates based on 2nd order HMMs
def GenerateFile(_model, _generate=True):
    m = _model
    permutations = {}
    if _generate:
        with open('generated.txt', 'w', encoding='utf-8') as f:
            for y0, y2 in m.y_y2.items():
                for prev_label, count in y2.items():
                    value = count/(m.y_count[prev_label])
                    permutations[(prev_label, y0)] = value
                    f.write("q({}|{})={}\n".format(y0, prev_label, value))
    else:
        for y0, y2 in m.y_y2.items():
            for prev_label, count in y2.items():
                value = count / (m.y_count[y0])
                permutations[(prev_label, y0)] = value

    return permutations


def GetTransitionDataFrame(_model):
    '''
    Part 4a - generating dataframe with all required transition params
    :param _model: from Model(file).train()
    :return: dataframe table
    '''

    perm_data = GenerateFile(_model, False)
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