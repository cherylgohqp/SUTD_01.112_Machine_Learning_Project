import unittest
import os
from Project.Cheryl_Project import document


class DocumentTest(unittest.TestCase):
    def testAddY(self):
        d = document.Document("")
        d.addToY("0")
        self.assertEqual(d.y["0"], 1)
        d.addToY("0")
        self.assertEqual(d.y["0"], 2)

    def testAddX(self):
        d = document.Document("")
        d.addToX("Hello", "Yes")
        d.addToX("World", "No")
        d.addToX("Hello", "Yes")
        d.addToX("Hello", "No")
        self.assertEqual(d.x, {
            "Hello": {
                "Yes": 2,
                "No": 1
            },
            "World": {
                "No": 1
            }
        })

    def testParse(self):
        d = document.Document("Cheryl_Project/testdata/small_test")  # this only works when calling from __main__
        d.parse()
        self.assertEqual(d.x, {
            'Hello': {'0': 1},
            'World': {'1': 1},
            'Not': {'b_positive': 1},
            'Really': {'a_positive': 1}
        })
        self.assertEqual(d.y, {
            '0': 1,
            '1': 1,
            'b_positive': 1,
            'a_positive': 1
        })

    def testFull(self):
        d = document.Document("Cheryl_Project/SG/train")
        d.parse()
        self.assertEqual(len(d.y), 7)
        d1 = document.Document("Cheryl_Project/CN/train")
        d1.parse()
        self.assertEqual(len(d1.y), 7)
        print(d.y)


if __name__ == '__main__':
    unittest.main()