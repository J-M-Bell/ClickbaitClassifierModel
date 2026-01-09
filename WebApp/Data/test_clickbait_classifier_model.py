from .clickbait_classifier_model import ClickbaitClassifierModel
import unittest

class ClickbaitClassifierModelTests(unittest.TestCase):
    """
    A test suite to test the ClickbaitClassifierModel methods.
    """
    def test_custom_tokenizer(self):
        """
        A unit test to test the output of the
        custom_tokenizer method in the ClickbaitClassifierModel
        
        :param self: ClickbaitClassifierModelTests - The ClickbaitClassifierModelTests object
        """

        #test if string is vectorized correctly
        model = ClickbaitClassifierModel()
        text = "Are You More Walter White Or Heisenberg"
        actual = model._custom_tokenizer(text)
        expected = ['you', 'walter', 'white', 'heisenberg']
        self.assertEqual(actual, expected)
    
    def test_predict(self):
        """
        A unit test to test the output of the
        predict method in the ClickbaitClassifierModel.
        
        :param self: ClickbaitClassifierModelTests - The ClickbaitClassifierModelTests object
        """

        # test if the prediction string is correct
        model = ClickbaitClassifierModel()
        text = "Are You More Walter White Or Heisenberg"
        actual = model.predict(text)
        expected = "clickbait"
        self.assertEqual(actual, expected)


# This allows you to run the tests directly from the file
if __name__ == '__main__':
    unittest.main()      