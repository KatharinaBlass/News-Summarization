from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
from nltk.stem import SnowballStemmer


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
        # note: download all modules before --> e.g. python -m spacy download fr_core_news_sm
        self.spacy_module_dict = {
            "de": "de_core_news_sm",
            "fr": "fr_core_news_sm",
            "es": "es_core_news_sm",
            "ru": "ru_core_news_sm",
            "tu": None,
            "en": "en_core_web_sm"
        }
        self.language = self.language_dict[lang]
        # no spacy model and no nltk stemmer available for turkish
        if self.language != 'turkish':
            spacy_module = self.spacy_module_dict[lang]
            self.nlp = spacy.load(spacy_module, disable=[
                                  'parser', 'tagger', 'ner'])
            self.stemmer = SnowballStemmer(language=self.language)
        else:
            self.nlp = None
            self.stemmer = None

    def tokenize_words(self, sent: str):
        return word_tokenize(sent, language=self.language)

    def tokenize_sents(self, text: str):
        return sent_tokenize(text, language=self.language)

    def get_stopwords(self):
        return stopwords.words(self.language)

    def lemmatize(self, sent):
        if self.nlp:
            sent = self.nlp(sent)
            return [word.lemma_.strip() for word in sent]
        else:
            return self.tokenize_words(sent)

    def stem(self, word):
        if self.stemmer:
            return self.stemmer.stem(word)
        else:
            return word
