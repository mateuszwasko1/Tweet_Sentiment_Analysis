import unittest
import os
import json
from project_name.preprocessing.ekphrasis_preprocessing import MainPreprocessing

class Preprocessing_Test(unittest.TestCase):
    def setUp(self):
        self.processor = MainPreprocessing()

    def test_apply_clean_text(self):
        pass
    def test_use_ekphrasis(self):
        pass
    def test_translate_emoji(self):
        text = "🙄😭😤"
        self.assertEqual(self.processor.translate_emoji(text), ":face_with_rolling_eyes::loudly_crying_face::face_with_steam_from_nose:")
        
    def test_remove_punctuation(self):
        text = "Now sizes S,XS(evenXXS sometimes) are too big, WTF?!"
        self.assertEqual(self.processor.remove_punctuation(text), "Now sizes S XSevenXXS sometimes are too big WTF?!")
    
