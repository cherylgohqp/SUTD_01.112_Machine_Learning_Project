import numpy as np
import pandas as pd


def Viterbi(_sentence, _model, _emission_df, _transition_df):
    """
    Takes in the sentence to be tagged, and returns the most likely sequence
    of labels
    :param _sentence: array of words to label
    :param _model: trained model to be used
    :param _emission_df: emission df from part 2
    :param _transition_df: transition df from part 3
    :return: None, generates file
    """

    # EXECUTE VITERBI
    # Exclude start and stop from states - will not be needing them
    states = [state for state, _ in _model.y_count.items()]
    states.remove("__START__")
    states.remove("__STOP__")

    # keep table of values:
    # (len(states) x len(sentence))
    basic_shape = np.zeros((len(states), len(_sentence))) * np.nan
    table1 = pd.DataFrame(basic_shape, index=states, columns=_sentence)

    # keep table of sequences
    sequence_shape = np.empty((len(states), len(_sentence)), dtype=list)
    seq_table = pd.DataFrame(sequence_shape, index=states, columns=_sentence)

    # base case - START to all states
    for i in range(len(states)):
        # transition prob from __START__ to anything
        transition_prob = _transition_df.loc['__START__', table1.index[i]]
        emission_prob = _emission_df.loc[_sentence[0], table1.index[i]]
        table1.iloc[i, 0] = float(transition_prob) * float(emission_prob)
        seq_table.iloc[i, 0] = ['__START__', states[i]]

    # iterative/recursive case - state to state
    for i in range(1, len(_sentence)):
        for j in range(len(states)):
            # find e(xi|yj)
            emission_prob = float(_emission_df.loc[_sentence[i], states[j]])

            # find the max transition prob from prev
            max_val = 0
            next_state_seq = []

            # from state to state prob
            for k in range(len(states)):
                prev_optimal = float(table1.iloc[k, i-1])
                prev_state_seq = seq_table.iloc[k, i-1]
                transition_prob = float(_transition_df.loc[states[k], states[j]])

                prob = prev_optimal * transition_prob * emission_prob
                if max_val == 0 or prob > max_val:
                    max_val = prob
                    next_state_seq = prev_state_seq + [states[k]]

            table1.iloc[j, i] = max_val
            seq_table.iloc[j, i] = next_state_seq

    # take optimal from table and return optimal val and sequence
    max_val = 0
    result_seq = []
    for i in range(len(states)):
        prob = float(table1.iloc[i, len(_sentence) - 1])  # take all from last
        if max_val == 0 or prob > max_val:
            max_val = prob
            result_seq = seq_table.iloc[i, len(_sentence) - 1]

    return result_seq[1:]


def Modified_Viterbi(_sentence, _model, _emission_df, _transition_df, _2nd_order_df):
    """
    Takes in the sentence to be tagged, and returns the most likely sequence
    of labels
    :param _sentence: array of words to label
    :param _model: trained model to be used
    :param _emission_df: emission df from part 2
    :param _transition_df: transition df from part 3
    :param _2nd_order_df: transition df from part 4
    :return: None, generates file
    """

    # EXECUTE VITERBI
    # Exclude start and stop from states - will not be needing them
    states = [state for state, _ in _model.y_count.items()]
    states.remove("__START__")
    states.remove("__STOP__")

    # keep table of values:
    # (len(states) x len(sentence))
    basic_shape = np.zeros((len(states), len(_sentence))) * np.nan
    table1 = pd.DataFrame(basic_shape, index=states, columns=_sentence)

    # keep table of sequences
    sequence_shape = np.empty((len(states), len(_sentence)), dtype=list)
    seq_table = pd.DataFrame(sequence_shape, index=states, columns=_sentence)

    # base case - START to all states, 1st order.
    # 2nd order not possible for base case
    for i in range(len(states)):
        # use 1st order, since 2nd order is non-existent
        transition_prob = _transition_df.loc['__START__', table1.index[i]]
        emission_prob = _emission_df.loc[_sentence[0], table1.index[i]]
        table1.iloc[i, 0] = float(transition_prob) * float(emission_prob)
        seq_table.iloc[i, 0] = ['__START__', states[i]]

    # iterative/recursive case - 2nd order
    for i in range(1, len(_sentence)):
        for j in range(len(states)):
            # find e(xi|yj)
            emission_prob = float(_emission_df.loc[_sentence[i], states[j]])

            # find the max transition prob from prev 2
            max_val = 0
            next_state_seq = []

            # from state to state prob
            for k in range(len(states)):
                prev_optimal = float(table1.iloc[k, i-1])
                prev_state_seq = seq_table.iloc[k, i-1]

                prev_1 = prev_state_seq[len(prev_state_seq) - 1]
                prev_2 = prev_state_seq[len(prev_state_seq) - 2]

                # use 2nd order here - modified
                transition_prob = float(_2nd_order_df.loc[states[k],
                                                          "({},{})".format(prev_2, prev_1)])

                prob = prev_optimal * transition_prob * emission_prob
                if max_val == 0 or prob > max_val:
                    max_val = prob
                    next_state_seq = prev_state_seq + [states[k]]

            table1.iloc[j, i] = max_val
            seq_table.iloc[j, i] = next_state_seq

    # take optimal from table and return optimal val and sequence
    max_val = 0
    result_seq = []
    for i in range(len(states)):
        prob = float(table1.iloc[i, len(_sentence) - 1])  # take all from last
        if max_val == 0 or prob > max_val:
            max_val = prob
            result_seq = seq_table.iloc[i, len(_sentence) - 1]

    return result_seq[1:]


def TagWithViterbi(_out, _file, _model, _emission_df, _transition_df, _2nd_order_df=None):
    """
    Takes in the file to be tagged and generates a new one
    :param _out: output file name
    :param _file: file with no labels
    :param _model: trained model to be used
    :param _emission_df: emission df from part 2
    :param _transition_df: transition df from part 3
    :param _2nd_order_df: 2nd order transition df from part 4, optional
    :return: None, generates file
    """

    # Generate array for possible words
    word_bag = _model.x_y_count
    reader = open(_file, 'r', encoding='utf-8')

    # Generate array of arrays for sentences in document
    unlabelled_tweets = []
    temp_data = []
    for line in reader:
        word = line.strip()
        word = word.lower()
        if word == "":
            unlabelled_tweets.append(temp_data)
            temp_data = []
        else:
            temp_data.append(word)
    unlabelled_tweets.append(temp_data)

    # Keep a global array of array of results for final
    # most likely states
    results = []

    # execute viterbi for each sentence
    for sentence in unlabelled_tweets:
        parsed_sentence = []
        # parse and replace unknowns with #UNK#
        for i in range(len(sentence)):
            if sentence[i] in word_bag:
                parsed_sentence.append(sentence[i])
            else:
                parsed_sentence.append('#UNK#')
        if _2nd_order_df is None:
            result = Viterbi(parsed_sentence, _model, _emission_df, _transition_df)
        else:
            result = Modified_Viterbi(parsed_sentence, _model, _emission_df, _transition_df, _2nd_order_df)
        results.append(result)

    # write results array into generated file
    writer = open(_out, 'w', encoding='utf-8')
    for i in range(len(unlabelled_tweets)):
        for j in range(len(unlabelled_tweets[i])):
            tweet = unlabelled_tweets[i][j]
            sentiment = results[i][j]
            writer.write('{} {}\n'.format(tweet, sentiment))
        writer.write('\n')  # empty line denoting end of tweet sentence
    writer.close()
    reader.close()
