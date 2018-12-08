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

    if not _sentence:
        return []

    # EXECUTE VITERBI
    states = [state for state, _ in _model.y_count.items()]

    # keep table of values
    # (len(states) x len(sentence))
    value_table = [[0 for x in range(len(_sentence) + 1)] for y in range(len(states))]

    # keep table of sequences
    sequence_table = [[[] for x in range(len(_sentence))] for y in range(len(states))]

    # base case - START to all states
    for i in range(len(states)):
        # transition prob from __START__ to anything
        try:
            transition_prob = _transition_df[('__START__', states[i])]
        except KeyError:
            transition_prob = 0.0

        # error occurs here due to empty _sentence
        try:
            emission_prob = _emission_df[(_sentence[0], states[i])]
        except KeyError:
            emission_prob = 0.0

        value_table[i][0] = float(transition_prob) * float(emission_prob)
        sequence_table[i][0] = ['__START__', states[i]]

    # iterative/recursive case - state to state
    for i in range(1, len(_sentence)):
        for j in range(len(states)):
            try:
                # find e(xi|yj)
                emission_prob = float(_emission_df[(_sentence[i], states[j])])
            except KeyError:
                emission_prob = 0.0

            # find the max transition prob from prev
            max_val = 0
            next_state_seq = []

            # from state to state prob
            for k in range(len(states)):
                prev_optimal = float(value_table[k][i-1])
                prev_state_seq = sequence_table[k][i-1]
                try:
                    transition_prob = float(_transition_df[(states[k], states[j])])
                except KeyError:
                    transition_prob = 0.0

                prob = prev_optimal * transition_prob * emission_prob
                if max_val == 0 or prob > max_val:
                    max_val = prob
                    next_state_seq = prev_state_seq + [states[k]]

            value_table[j][i] = max_val
            sequence_table[j][i] = next_state_seq

        # end case - all states to __STOP__
        for i in range(len(states)):
            try:
                transition_prob = _transition_df[(states[i], '__STOP__')]
            except KeyError:
                transition_prob = 0.0

            value_table[i][-1] = float(transition_prob) * float(value_table[i][-2])

    # take optimal from table and return optimal val and sequence
    max_val = 0
    result_seq = []
    for i in range(len(states)):
        prob = float(value_table[i][-1])  # take all from last
        if max_val == 0 or prob > max_val:
            max_val = prob
            result_seq = sequence_table[i][-1]

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

    if not _sentence:
        return []

    # EXECUTE VITERBI
    states = [state for state, _ in _model.y_count.items()]

    # keep table of values
    # (len(states) x len(sentence))
    value_table = [[0 for x in range(len(_sentence) + 1)] for y in range(len(states))]

    # keep table of sequences
    sequence_table = [[[] for x in range(len(_sentence))] for y in range(len(states))]


    # base case - START to all states, 1st order.
    # 2nd order not possible for base case
    for i in range(len(states)):
        # use 1st order, since 2nd order is non-existent
        # transition prob from __START__ to anything
        try:
            transition_prob = _transition_df[('__START__', states[i])]
        except KeyError:
            transition_prob = 0.0

        # error occurs here due to empty _sentence
        try:
            emission_prob = _emission_df[(_sentence[0], states[i])]
        except KeyError:
            emission_prob = 0.0

        value_table[i][0] = float(transition_prob) * float(emission_prob)
        sequence_table[i][0] = ['__START__', states[i]]

    # iterative/recursive case - 2nd order
    for i in range(1, len(_sentence)):
        for j in range(len(states)):
            try:
                # find e(xi|yj)
                emission_prob = float(_emission_df[(states[j], _sentence[i])])
            except KeyError:
                emission_prob = 0

            # find the max transition prob from prev 2
            max_val = 0
            next_state_seq = []

            # from state to state prob
            for k in range(len(states)):
                prev_optimal = float(value_table[k][i-1])
                prev_state_seq = sequence_table[k][i-1]

                prev_1 = prev_state_seq[len(prev_state_seq) - 1]
                prev_2 = prev_state_seq[len(prev_state_seq) - 2]

                # use 2nd order here - modified
                try:
                    transition_prob = float(_2nd_order_df[((prev_2, prev_1), states[k])])
                except KeyError:
                    transition_prob = 0.0

                prob = prev_optimal * transition_prob * emission_prob
                if max_val == 0 or prob > max_val:
                    max_val = prob
                    next_state_seq = prev_state_seq + [states[k]]

            value_table[j][i] = max_val
            sequence_table[j][i] = next_state_seq

    # end case - all states to __STOP__
    for i in range(len(states)):

        prev_optimal = float(value_table[i][-2])
        prev_state_seq = sequence_table[i][-1]
        prev_1 = prev_state_seq[len(prev_state_seq) - 1]
        prev_2 = prev_state_seq[len(prev_state_seq) - 2]
        try:
            transition_prob = float(_2nd_order_df[((prev_2, prev_1), '__STOP__')])
        except KeyError:
            transition_prob = 0.0

        value_table[i][-1] = float(transition_prob) * float(value_table[i][-2])

    # take optimal from table and return optimal val and sequence
    max_val = 0
    result_seq = []
    for i in range(len(states)):
        prob = float(value_table[i][-1])  # take all from last
        if max_val == 0 or prob > max_val:
            max_val = prob
            result_seq = sequence_table[i][-1]

    return result_seq[1:]


def TagWithViterbi(_out, _file, _model, _emission_df, _transition_df, _2nd_order_df=None):
    """
    Takes in the file to be tagged and generates a new one
    NOTE THAT ALL DFs HAVE BEEN CHANGED TO HASHMAPS
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
            if temp_data:  # catch any multiple line breaks
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
