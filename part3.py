import numpy as np
import pandas as pd


# TRY NOT TO USE THIS
def GenerateFile(_model, _generate=False):
    """
    Aims to create a dict to give transition probabilities
    :param _model: model to be used in generating this temp dict
    :param _generate: Bool indicating if a file needs to be generated, usually False
    :return: Dict in the form of { (yi-1, yi): P(yi,yi-1) }
    """
    m = _model
    permutations = {}
    if _generate:
        with open('generated.txt', 'w', encoding='utf-8') as f:
            for y0, y1 in m.y_y1.items():
                for prev_label, count in y1.items():
                    value = count/(m.y_count[y0])
                    permutations[(prev_label, y0)] = value
                    f.write("q({}|{})={}\n".format(y0, prev_label, value))
    else:
        for yi, y1 in m.y_y1.items():
            for yi_1, count in y1.items():  # count = Count(yi, yi-1)
                value = count / (m.y_count[yi_1])  # P(yi|yi-1) = Count(yi,yi-1)/Count(yi-1)
                permutations[(yi_1, yi)] = value

    return permutations


def GetTransitionDataFrame(_model):
    '''
    Part a - generating dataframe with all required transition params
    :param _model: from Model(file).train()
    :return: dataframe table
    '''

    perm_data = GenerateFile(_model, False)
    labels = ['__START__', 'O', 'B-positive', 'I-positive', '__STOP__',
              'B-negative', 'I-negative', 'B-neutral', 'I-neutral']

    data = [[''] + labels]
    for label1 in labels:
        add_data = [label1]
        for label2 in labels:
            try:
                add_data.append(perm_data[(label1, label2)])
            except KeyError:
                add_data.append('0')

        data.append(add_data)

    # Create dataframe
    df = np.array(data)
    return pd.DataFrame(data=df[1:, 1:],
                        index=df[1:, 0],
                        columns=df[0, 1:])


# # NOT FOR USE
# def writeDFToFile(name, df):
#     with open(name, 'w', encoding='utf-8') as f:
#         with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#             f.write(str(df))
#
#
# def part3a():
#     m = Model('SG/train')
#     m.train()
#     writeDFToFile('SG_Pretty_print_df.txt', GetTransitionDataFrame(m))





