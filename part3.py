def GetTransitionDataFrame(_model):
    """
    Aims to create a dict to give transition probabilities
    :param _model: model to be used in generating this temp dict
    :param _generate: Bool indicating if a file needs to be generated, usually False
    :return: Dict in the form of { (yi-1, yi): P(yi,yi-1) }
    """

    m = _model
    permutations = {}
    for yi, y1 in m.y_y1.items():
        for yi_1, count in y1.items():  # count = Count(yi, yi-1)
            value = count / (m.y_count[yi_1])  # P(yi|yi-1) = Count(yi,yi-1)/Count(yi-1)
            permutations[(yi_1, yi)] = value

    return permutations







