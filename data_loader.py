import nltk
from datasets import load_dataset


class DataLoader():
    def __init__(self, language="de"):
        if language == "en":
            self.corpus_loader = CNNCorpusLoader()
        else:
            self.corpus_loader = MLSUMCorpusLoader(language)

    def split_sentences(self, text: str):
        return nltk.tokenize.sent_tokenize(text)

    def get_formatted_data(self, data_type: str = "test"):
        return self.corpus_loader.get_formatted_data(data_type)


class MLSUMCorpusLoader(DataLoader):
    def __init__(self, language):
        self.dataset = load_dataset("mlsum", language)

    def get_formatted_data(self, type: str):
        data = self.dataset[type]
        articles = [
            self.split_sentences(article["text"]) for article in data]
        summaries = [
            self.split_sentences(article["summary"]) for article in data]
        headlines = [
            self.split_sentences(article["title"]) for article in data]

        return {
            "articles": articles, "summaries": summaries, "headlines": headlines
        }


class CNNCorpusLoader(DataLoader):
    def __init__(self):
        self.dataset = load_dataset("cnn_dailymail", "3.0.0")

    def get_formatted_data(self, type: str):
        data = self.dataset[type]
        articles = [
            self.split_sentences(article["article"]) for article in data]
        summaries = [
            self.split_sentences(article["highlights"]) for article in data]

        return {
            "articles": articles, "summaries": summaries, "headlines": None
        }
