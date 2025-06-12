import unittest
from tweet_sentiment_analysis.preprocessing.bert_preprocessing import (
    MainPreprocessing)


class Preprocessing_Test(unittest.TestCase):
    def setUp(self):
        self.processor = MainPreprocessing()

    def test_translate_emoji(self):
        text = "🙄😭😤"
        self.assertEqual(
            self.processor._translate_emoji(text),
            (":face_with_rolling_eyes::loudly_crying_face::"
             "face_with_steam_from_nose:"))

    def test_whole_preprocessing(self):
        text = "@user     Are you 😢?! &amp; #Happy https://google.com ***"
        self.assertEqual(self.processor.clean_text(text),
                         "Are you crying face ?! Happy")
