import unittest

from main import hello_world


class MainTest(unittest.TestCase):
    """
    Unit tests for the main module functions.
    """

    def test_hello(self) -> None:
        """
        Verify that hello_world returns the expected greeting string.
        """
        expected: str = "Hello, World!"
        result: str = hello_world()
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
