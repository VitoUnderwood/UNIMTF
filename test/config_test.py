import unittest
from configs.PrefixNlgConfig import PrefixNlgConfig

class MyTestCase(unittest.TestCase):
    def test_something(self):
        config = PrefixNlgConfig.getParser()
        print(config.__dict__)


if __name__ == '__main__':
    unittest.main()
