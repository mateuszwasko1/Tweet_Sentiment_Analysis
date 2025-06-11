import unittest
from project_name.preprocessing.bert_preprocessing import MainPreprocessing

"""
Unit tests for the MainPreprocessing class, verifying
text cleaning, tokenization, emoji translation, and
punctuation removal functionalities.
"""


class PreprocessingTest(unittest.TestCase):
    """
    Test suite for MainPreprocessing methods using
    Ekphrasis-based text transformations.
    """
    def setUp(self) -> None:
        """
        Initialize MainPreprocessing instance for each test.
        """
        self.processor: MainPreprocessing = MainPreprocessing()

    def test_apply_clean_text_whitespace(self) -> None:
        """
        Leading, trailing, and multiple intermediate
        spaces should be collapsed.
        """
        text: str = "i like that     really"
        expected: list[str] = ["i", "like", "that", "really"]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_ekphrasis_user(self) -> None:
        """
        User mentions should be replaced with <user> token.
        """
        text: str = "@friend i can not believe that"
        expected: list[str] = ["<user>", "i", "can", "not", "believe", "that"]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_ekphrasis_email(self) -> None:
        """
        Email addresses should be replaced with <email> token.
        """
        text: str = (
            "@friend i can not believe that please email me at test@ex.com"
        )
        expected: list[str] = [
            "<user>", "i", "can", "not", "believe", "that",
            "please", "email", "me", "at", "<email>"
        ]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_ekphrasis_url(self) -> None:
        """
        URLs should be replaced with <url> token.
        """
        text: str = "please visit https://www.example.com"
        expected: list[str] = ["please", "visit", "<url>"]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_ekphrasis_hashtag_segmenter(self) -> None:
        """
        Hashtags should be wrapped and segmented into words.
        """
        text: str = "it is the #BestDayEver"
        expected: list[str] = [
            "it", "is", "the", "<hashtag>", "best", "day", "ever",
            "</hashtag>"
        ]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_ekphrasis_allcaps(self) -> None:
        """
        All-caps words should be wrapped with <allcaps> markers.
        """
        text: str = "it is THE Best Day Ever"
        expected: list[str] = [
            "it", "is", "<allcaps>", "the", "</allcaps>", "best",
            "day", "ever"
        ]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_ekphrasis_repeated_punctuation(self) -> None:
        """
        Repeated punctuation should yield a single punctuation token
        and <repeated> marker.
        """
        text: str = "it is the best day ever!!!!"
        expected: list[str] = ["it", "is", "the", "best", "day",
                               "ever", "!", "<repeated>"]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_ekphrasis_emphasis_and_elongated(self) -> None:
        """
        Emphasized words and elongated characters should be marked accordingly.
        """
        text: str = "it is the *best* day ever coool"
        expected: list[str] = [
            "it", "is", "the", "best", "<emphasis>", "day", "ever", "cool",
            "<elongated>"
        ]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_ekphrasis_censored(self) -> None:
        """
        Censored words should be detected and marked.
        """
        text: str = "it is the best day ever f**k"
        expected: list[str] = ["it", "is", "the", "best", "day", "ever",
                               "f**k", "<censored>"]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_ekphrasis_contractions(self) -> None:
        """
        Contractions should be expanded into constituent words.
        """
        text: str = "i can't believe that"
        expected: list[str] = ["i", "can", "not", "believe", "that"]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_ekphrasis_emoticons(self) -> None:
        """
        Common emoticons should be replaced with emotion tokens.
        """
        text: str = "i like that :)"
        expected: list[str] = ["i", "like", "that", "<happy>"]
        self.assertEqual(
            self.processor.use_ekphrasis(text),
            expected
        )

    def test_translate_emoji(self) -> None:
        """
        Emojis should be translated into descriptive colon-wrapped strings.
        """
        text: str = "🙄😭😤"
        expected: str = (
            ":face_with_rolling_eyes::loudly_crying_face::"
            "face_with_steam_from_nose:"
        )
        self.assertEqual(
            self.processor.translate_emoji(text),
            expected
        )

    def test_remove_punctuation(self) -> None:
        """
        Punctuation tokens should be removed except
        for question and exclamation marks.
        """
        tokens: list[str] = ["i", "like", "that", ".", "very", "much", "?",
                             "!", "."]
        expected: list[str] = ["i", "like", "that", "very", "much", "?", "!"]
        self.assertEqual(
            self.processor.remove_punctuation(tokens),
            expected
        )


if __name__ == "__main__":
    unittest.main()
