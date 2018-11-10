import string


class Document:
    def __init__(self, _file, _document=None):
        self.file = _file
        self.document = _document
        self.k = 0
        self.x = {}  # Will contain dictionary of words with labels
        self.y = {}  # contains all labels with number of occurances

    def addToY(self, _label):
        if _label not in self.y:
            self.y[_label] = 1
        else:
            self.y[_label] += 1

    def addToX(self, _word, _label):
        if _word not in self.x:
            self.x[_word] = {_label: 1}
        else:
            w = self.x[_word]
            if _label not in w:
                w[_label] = 1
            else:
                w[_label] += 1

    def handleEdges(self, _part):
        if len(_part) > 2:
            for i in range(1, len(_part) - 1):
                _part[0] += " "
                _part[0] += _part[i]
            _part[1] = _part[len(_part) - 1]
        return _part

    def parse(self):
        with open(self.file, encoding='utf-8') as f:
            for line in f:
                part = line.strip().split(" ")
                part = self.handleEdges(part)
                try:
                    self.addToY(part[1])
                    self.addToX(part[0], part[1])
                except:
                    # likely this is caused by an empty string
                    continue

    # Objective of this is to read a file with only words
    # returns number of words not in test
    # WARNING: This will give duplicates, intended.
    def parseWithTrainedData(self):
        if self.document is None:
            print("Cannot parse None document")
            return None
        with open(self.file, encoding='utf-8') as f:
            for line in f:
                part = line.strip()
                if part not in self.document.x:
                    self.k += 1
        return self.k


