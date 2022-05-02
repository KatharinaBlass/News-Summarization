import nltk
from datasets import load_dataset


class DataLoader():
    def __init__(self, language="de"):
        self.dataset = load_dataset("mlsum", language)

    def split_sentences(self, text: str):
        return nltk.tokenize.sent_tokenize(text)

    def get_formatted_data(self, type: str = "test"):
        data = self.dataset[type]
        articles = [
            self.split_sentences(article["text"]) for article in data]
        summaries = [
            self.split_sentences(article["summary"]) for article in data]

        return {
            "articles": articles, "summaries": summaries
        }
