import unittest
import os
from Project.Cheryl_Project import model


class ModelTest(unittest.TestCase):
    def testAddY(self):
        d = model.Model("")
        d.addToY("0")
        self.assertEqual(d.y_count["0"], 1)
        d.addToY("0")
        self.assertEqual(d.y_count["0"], 2)

    def testAddX(self):
        d = model.Model("")
        d.addToX("Hello", "Yes")
        d.addToX("World", "No")
        d.addToX("Hello", "Yes")
        d.addToX("Hello", "No")
        self.assertEqual(d.x_y_count, {
            "Hello": {
                "Yes": 2,
                "No": 1
            },
            "World": {
                "No": 1
            }
        })

    def testParse(self):
        d = model.Model("testdata/small_test")
        d.train()
        print(d.x_y_count)
        print(d.y_count)
        self.assertEqual(d.x_y_count, {
            'hello': {'0': 1},
            'world': {'1': 1},
            'not': {'b_positive': 1},
            'really': {'a_positive': 1}
        })
        self.assertEqual(d.y_count, {
            '__START__': 1,
            '0': 1,
            '1': 1,
            'b_positive': 1,
            'a_positive': 1
        })

    def testFull(self):
        d = model.Model("SG/train")
        d.train()
        self.assertEqual(len(d.y_count), 9)
        d1 = model.Model("CN/train")
        d1.train()
        self.assertEqual(len(d1.y_count), 9)
        print(d.y_count)
        unique_words = len(d.x_y_count)
        print(unique_words, "number of unique words")

        '''
        Verify same number of words have been recorded
        '''
        word_count = 0
        for _, word_labels in d.x_y_count.items():
            for _, word_label_count in word_labels.items():
                word_count += word_label_count

        final_count = 0
        for token, count in d.y_count.items():
            if token != "__START__" and token != "__STOP__":
                final_count += count
        self.assertEqual(final_count, word_count, "They should have the same count")

        print(len(d.y_y1), "number of unique prev states")


if __name__ == '__main__':
    unittest.main()