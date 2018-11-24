from model import Model
import numpy as np
import pandas as pd


def GetEmissionDataFrame(_model, _k=0):
    '''
    This is for external usage as API,
    also part 2a)
    if k > 0, this will be used for part 2b)
    :param _model: model from Model
    :param _k: k number to be classified as #UNK#
    :return: dataframe
    '''

    m = _model

    # create emissions data
    emissions = {}
    for word, labels in m.x_y_count.items():
        for label, count in labels.items():
            if _k > 0 and count <= _k:
                # UNK token here
                word = '#UNK#'
                emissions[(word, label)] = _k / (m.y_count[label] + _k)
            else:
                emissions[(word, label)] = count/m.y_count[label]

    # create dataframe from emissions data
    labels = ['__START__', 'O', 'B-positive', 'I-positive', '__STOP__',
              'B-negative', 'I-negative', 'B-neutral', 'I-neutral']
    data = [[''] + labels]

    for word, _ in m.x_y_count.items():
        add_data = [word]
        for label in labels:
            try:
                add_data.append(emissions[(word, label)])
            except KeyError:
                add_data.append('0')

        data.append(add_data)

    # create dataframe from data
    arr = np.array(data)
    return pd.DataFrame(data=arr[1:,1:],
                        index=arr[1:, 0],
                        columns=arr[0, 1:])


def findMax(_df_row):
    max_label = 'O'
    max_val = 0.00
    for label, val in _df_row.items():
        if float(val) > max_val:
            max_label = label
            max_val = float(val)
    return max_label, max_val


def TagTweets(_emission_df, _file):
    '''
    This takes in a dataframe and a file, and tags
    all the words in the file to sentiment tweets
    :param _emission_df: emission dataframe from above
    :param _file: must be a valid file without sentiments
    :return: None, generates another file with sentiments filled
    '''
    reader = open(_file, 'r', encoding='utf-8')
    writer = open(_file + "_generated.txt", 'w', encoding='utf-8')
    for line in reader:
        word = line.strip()
        try:
            word_row = _emission_df.loc[word]
            max_label, _ = findMax(word_row)
        except KeyError:
            if word == '':
                max_label = ''
            else:
                max_label = 'O'
        finally:
            writer.write("{} {}\n".format(word, max_label))


if __name__ == '__main__':
    files = ['SG', 'EN', 'FR', 'CN']
    for f in files:
        m = Model(f + '/train')
        m.train()
        df = GetEmissionDataFrame(m, 1)
        TagTweets(df, f + '/dev.in')






