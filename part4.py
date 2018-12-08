# This generates based on 2nd order HMMs
def GetTransitionDataFrame(_model):
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
            # P(yi | yi-2,yi-1) = Count(yi,yi-1,yi-2) / Count(yi-2, yi-1)
            value = count / (m.y_y1[yi_1][yi_2])
            # will save in the form { ((yi-2,yi-1),yi): value }
            permutations[(prev_labels, yi)] = value

    return permutations

