import unittest
from project_name.preprocessing.ekphrasis_preprocessing import (
    MainPreprocessing)


class Preprocessing_Test(unittest.TestCase):
    def setUp(self):
        self.processor = MainPreprocessing()

    def test_apply_clean_text_whitespace(self):
        text = "i like that     really"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['i', 'like', 'that', 'really']))

    def test_ekphrasis_user(self):
        text = "@friend i can not believe that"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['<user>', 'i', 'can', 'not', 'believe', 'that',]
                    ))

    def test_ekphrasis_email(self):
        text = "@friend i can not believe that please email me at test@ex.com"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['<user>', 'i', 'can', 'not', 'believe', 'that', 'please',
             'email', 'me', 'at', '<email>']
                    ))

    def test_ekphrasis_url(self):
        text = "please visit https://www.example.com"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['please', 'visit', '<url>']))

    def test_ekphrasis_hashtag_segmenter(self):
        text = "it is the #BestDayEver"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['it', 'is', 'the', '<hashtag>', 'best',
             'day', 'ever', '</hashtag>']))

    def test_ekphrasis_allcaps(self):
        text = "it is THE Best Day Ever"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['it', 'is', '<allcaps>', 'the', '</allcaps>', 'best',
             'day', 'ever']))

    def test_ekphrasis_repeated_punctuation(self):
        text = "it is the best day ever!!!!"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['it', 'is', 'the', 'best',
             'day', 'ever', '!', '<repeated>']))

    def test_ekphrasis_emphasis_and_elongated(self):
        text = "it is the *best* day ever coool"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['it', 'is', 'the', 'best', '<emphasis>', 'day', 'ever',
                'cool', '<elongated>']))

    def test_ekphrasis_censored(self):
        text = "it is the best day ever f**k"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['it', 'is', 'the', 'best', 'day', 'ever', 'f**k', '<censored>']))

    def test_ekphrasis_contractions(self):
        text = "i can't believe that"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['i', 'can', 'not', 'believe', 'that']))

    def test_ekphrasis_emoticons(self):
        text = "i like that :)"
        self.assertEqual(self.processor.use_ekphrasis(text), (
            ['i', 'like', 'that', '<happy>']))

    """
    def test_ekphrasis_spelling(self):
        text = "i can't thier that"
        self.assertEqual(self.processor.use_ekphrasis(text), (
                    ['i', 'can', 'not', 'their', 'that']))
    """

    def test_translate_emoji(self):
        text = "🙄😭😤"
        self.assertEqual(
            self.processor.translate_emoji(text),
            (":face_with_rolling_eyes::loudly_crying_face::"
             "face_with_steam_from_nose:"))

    def test_remove_punctuation(self):
        tokens = ['i', 'like', 'that', '.', 'very', 'much', '?', '!', '.']
        self.assertEqual(self.processor.remove_punctuation(tokens), (
            ['i', 'like', 'that', 'very', 'much', '?', '!']))
