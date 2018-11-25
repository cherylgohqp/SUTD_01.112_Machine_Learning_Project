'''
Model is used to accept training data and parse it
The final results of training should be:
y_count: { label:count }
x_y_count: { word: {label:count} }
y_y1: {label: {prev_label:count} }
y_y2: {label: {prev_prev_label:count} }
use Model("SG/train").train() to train it and return a model
'''


class Model:
    def __init__(self, _file):
        self.file = _file
        self.x_y_count = {}  # Will contain dictionary of words with labels
        self.y_count = {}  # contains all labels with number of occurances
        '''
        Add number of dicts accordingly per order HMM desired
        '''
        self.prev_y1 = ""
        self.prev_y2 = ""
        self.y_y1 = {}
        self.y_y2 = {}

    def addToY(self, _label):
        if _label not in self.y_count:
            self.y_count[_label] = 1
        else:
            self.y_count[_label] += 1

    def addToX(self, _sentiment, _label):
        if _sentiment not in self.x_y_count:
            self.x_y_count[_sentiment] = {_label: 1}
        else:
            w = self.x_y_count[_sentiment]
            if _label not in w:
                w[_label] = 1
            else:
                w[_label] += 1

    def addPrevY(self, _label):
        # label is the current word/token being processed
        if _label not in self.y_y1:
            self.y_y1[_label] = {}
        if self.prev_y1 != "":
            if self.prev_y1 not in self.y_y1[_label]:
                self.y_y1[_label][self.prev_y1] = 1
            else:
                self.y_y1[_label][self.prev_y1] += 1
        if _label not in self.y_y2:
            self.y_y2[_label] = {}
        if self.prev_y2 != "":
            if (self.prev_y2, self.prev_y1) not in self.y_y2[_label]:
                self.y_y2[_label][(self.prev_y2, self.prev_y1)] = 1
            else:
                self.y_y2[_label][(self.prev_y2, self.prev_y1)] += 1

    # This should only be called when training data,
    # changes prev y and prev prev y
    def changeState(self, _label=""):
        self.addPrevY(_label)
        self.prev_y2 = self.prev_y1
        self.prev_y1 = _label

    def handleEdges(self, _part):
        if len(_part) > 2:
            for i in range(1, len(_part) - 1):
                _part[0] += " "
                _part[0] += _part[i]
            _part[1] = _part[len(_part) - 1]
        return _part

    def train(self):
        '''
        This should be the main method called to parse a document
        '''
        with open(self.file, encoding='utf-8') as f:
            for line in f:
                part = line.strip().split(" ")
                part = self.handleEdges(part)
                try:
                    part[0] = part[0].lower()
                    token = part[1]
                    if self.prev_y1 == "__STOP__" or self.prev_y1 == "":
                        # restart
                        token = "__START__"
                        self.prev_y2 = ""
                        self.prev_y1 = ""
                    else:
                        self.addToX(part[0], part[1])
                    self.addToY(token)
                    self.changeState(token)
                except:
                    # likely this is caused by an empty string
                    # Treat as STOP
                    token = "__STOP__"
                    self.addToY(token)
                    self.changeState(token)




