import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import emoji
from cleantext import clean
#from punctuation import Punctuation

class MainPreprocessing():
    def __init__(self,test_data: bool = False):
        self.test_data = test_data
        self.processor = TextPreProcessor(normalize=[
            'url', 'email', 'percent', 'money', 'phone', 'user',
            'time', 'date', 'number'], annotate={"hashtag", "allcaps", 
            "elongated", "repeated", 'emphasis', 'censored'}, segmenter="twitter", 
            corrector= "twitter", unpack_contractions=True, spell_correct_elong=True, 
            unpack_hashtags= True, dicts= [emoticons],
            tokenizer = SocialTokenizer(lowercase = True).tokenize)
        
    def extract_features_labels(self, df: pd.DataFrame, feature_name: str,
                                label_name: str) -> tuple[pd.DataFrame,
                                                          pd.DataFrame]:
        # Add value error to check if allowed names are passed on
        X = df[feature_name]
        y = df[label_name]
        return X, y

    def apply_clean_text(self, text: str) -> str: 
        return clean( text,
        to_ascii = True,
        lower = True,
        normalize_whitespace = True,
        no_line_breaks = False,
        strip_lines = True,
        keep_two_line_breaks = False,
        no_urls = False,
        no_emails = False,
        no_phone_numbers = False,
        no_numbers = False,
        replace_with_currency_symbol= "<currency>",
        replace_with_digit= "<digit>",
        replace_with_email="<user>",
        replace_with_number="<num>",
        replace_with_url= "<url>")
    
    def use_ekphrasis(self, text:str) -> str:
        return self.processor.pre_process_doc(text)
    
    def translate_emoji(self, text:str) -> str:
        return emoji.demojize(text)
    
    def clean_text(self,text:str)->str:
        text = self.translate_emoji(text)
        tokens = self.use_ekphrasis(text)
        tokens = [self.apply_clean_text(token) for token in tokens]
        return tokens
    """
    def vectorize(self, X: pd.DataFrame, use_tfidf: bool = True):
        vectorizer = TfidfVectorizer()
        if use_tfidf is True:
            if self.test_data is True:
                X_vec = vectorizer.transform(X)
            else:
                X_vec = vectorizer.fit_transform(X)
        # else: Glove Implementation
        return X_vec
    """
    
    def preprocess_df(self, df: pd.DataFrame):
        X, y = self.extract_features_labels(df, "tweet", "emotion")
        X = X.apply(self.clean_text)
        #X_vec = self.vectorize(X, use_tfidf=True)
        #return X_vec, y
        return X,y

    def preprocessing_pipeline(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(script_dir, "..", "..", "data", "raw",
                                  "training_merged.json")
        dev_path = os.path.join(script_dir, "..", "..", "data", "raw",
                                "development_merged.json")
        test_path = os.path.join(script_dir, "..", "..", "data", "raw",
                                 "test-gold_merged.json")
        training_data = pd.read_json(train_path, orient="records", lines=True)
        dev_data = pd.read_json(dev_path,
                                orient="records", lines=True)
        test_data = pd.read_json(test_path,
                                 orient="records", lines=True)
        X_training, y_training = self.preprocess_df(training_data)
        X_dev, y_dev = self.preprocess_df(dev_data)
        X_test, y_test = self.preprocess_df(test_data)
        return (X_training, y_training), (X_dev, y_dev), (X_test, y_test)
    
if __name__ == "__main__":
    preprocessor = MainPreprocessing()
    (X_trainign,y_training), (X_dev, y_dev), (X_test, y_test) = preprocessor.preprocessing_pipeline()

    print(X_trainign[7])