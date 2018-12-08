import numpy as np


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
                if (word, label) not in emissions:
                    emissions[(word, label)] = 0
                emissions[(word, label)] += _k / (m.y_count[label] + _k)
            else:
                emissions[(word, label)] = count/m.y_count[label]
    for wl, val in emissions.items():
        if wl[0] == '#UNK#':
            m.x_y_count['#UNK#'] = {}
            m.x_y_count['#UNK#'][wl[1]] = val

    # # create dataframe from emissions data
    # states = [state for state, _ in _model.y_count.items()]
    # words = [word for word, _ in _model.x_y_count.items()]
    #
    # # create dataframe
    # basic_shape = np.zeros((len(states), len(words)))
    # df = pd.DataFrame(basic_shape, index=states, columns=words)
    #
    # for w in words:
    #     for s in states:
    #         # fill up column by column
    #         try:
    #             df.loc[s, w] = emissions[(w, s)]
    #         except KeyError:
    #             df.loc[s, w] = 0.0

    return emissions


def findMax(_emission_df, word):
    # print("finding max of:", word)

    values_row = [(key, v) for key, v in _emission_df.items() if key[0] == word]

    if not values_row:
        values_row = [(key, v) for key, v in _emission_df.items() if key[0] == "#UNK#"]

    # sort based on value and return first (max)
    return sorted(values_row, key=lambda x: x[1], reverse=True)[0]


def TagTweets(_out, _emission_df, _file):
    '''
    This takes in a dataframe and a file, and tags
    all the words in the file to sentiment tweets
    :param _out: file to be generated
    :param _emission_df: emission dataframe from above
    :param _file: must be a valid file without sentiments
    :return: None, generates another file with sentiments filled
    '''
    reader = open(_file, 'r', encoding='utf-8')
    writer = open(_out, 'w', encoding='utf-8')
    for line in reader:
        word = line.strip()

        max_tuple = findMax(_emission_df, word)  # (('word','sentiment'), value)

        max_label = max_tuple[0][1]
        writer.write("{} {}\n".format(word, max_label))


# if __name__ == '__main__':
#     files = ['SG', 'EN', 'FR', 'CN']
#     for f in files:
#         m = Model(f + '/train')
#         m.train()
#         df = GetEmissionDataFrame(m)
#         TagTweets(df, f + '/dev.in')







