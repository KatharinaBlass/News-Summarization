import pandas as pd
import nltk


class DataLoader():
    def __init__(self):
        #self.train_data = self.load_corpus("./data/train.csv")
        #self.validation_data = self.load_corpus("./data/validation.csv")
        self.test_data = self.load_corpus("./data/test.csv")

    def split_sentences(self, text: str):
        return nltk.tokenize.sent_tokenize(text)

    def load_corpus(self, filepath: str):
        news = pd.read_csv(
            filepath,
            usecols=["article", "highlights"],
        )
        articles_with_tokenized_sents = news.article.apply(
            self.split_sentences)
        highlights_with_tokenized_sents = news.highlights.apply(
            self.split_sentences)

        return {
            "articles": articles_with_tokenized_sents, "highlights": highlights_with_tokenized_sents
        }
