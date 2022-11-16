import sys
import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        print(sys.path)


if __name__ == '__main__':
    unittest.main()
