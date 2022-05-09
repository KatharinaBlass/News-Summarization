from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


class Helper():
    def __init__(self, lang):
        self.language_dict = {
            "de": 'german',
            "fr": 'french',
            "es": 'spanish',
            "ru": 'russian',
            "tu": 'turkish',
            "en": 'english'
        }
        self.language = self.language_dict[lang]

    def tokenize_words(self, sent: str):
        return word_tokenize(sent, language=self.language)

    def tokenize_sents(self, text: str):
        return sent_tokenize(text, language=self.language)

    def get_stopwords(self):
        return stopwords.words(self.language)
