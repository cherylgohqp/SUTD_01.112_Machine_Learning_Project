from model import Model
import numpy as np
import pandas as pd



def trainModel(file):
    m = Model(file)
    m.train()
    return m


def GenerateFile(_model, _generate=True):
    m = _model
    permutations = {}
    if _generate:
        with open('generated.txt', 'w', encoding='utf-8') as f:
            for y0, y1 in m.y_y1.items():
                for prev_label, count in y1.items():
                    value = count/(m.y_count[prev_label])
                    permutations[(prev_label, y0)] = value
                    f.write("q({}|{})={}\n".format(y0, prev_label, value))
    else:
        for y0, y1 in m.y_y1.items():
            for prev_label, count in y1.items():
                value = count / (m.y_count[prev_label])
                permutations[(prev_label, y0)] = value

    return permutations


def generateDataframe(_model):
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
    return pd.DataFrame(data=df[1:,1:],
                        index=df[1:, 0],
                        columns=df[0, 1:])


def writeDFToFile(name, df):
    with open(name, 'w', encoding='utf-8') as f:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            f.write(str(df))


def part3a():
    m = trainModel('SG/train')
    writeDFToFile('SG_Pretty_print_df.txt', generateDataframe(m))


def viterbi(_file):
    '''
    part b - viterbi algorithm on dev.in
    :param _file: for dev.in file
    :return:
    '''
    return


# if __name__ == "__main__":
#     m = trainModel('SG/train')
#     writeDFToFile('Pretty_print_df.txt', generateDataframe(m))
