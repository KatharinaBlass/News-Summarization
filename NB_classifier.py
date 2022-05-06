from basic_summarizer import BasicSummarizer
import numpy
import nltk
from nltk.classify import NaiveBayesClassifier


class NaiveBayesSummarizer(BasicSummarizer):
    def __init__(self, train_articles: list[list[str]], train_labels: list[numpy.ndarray], train_headlines: list, validate_articles: list[list[str]], validate_labels: list[numpy.ndarray], validate_headlines: list):
        super().__init__()
        self.train_data = list(zip(self.merge_articles(
            train_articles), self.get_article_sent_position(train_articles), self.get_article_headline(train_articles, train_headlines), self.merge_labels(train_labels)))
        self.validation_data = list(zip(self.merge_articles(
            validate_articles), self.get_article_sent_position(validate_articles), self.get_article_headline(validate_articles, validate_headlines), self.merge_labels(validate_labels)))
        self.classifier = NaiveBayesClassifier
        self.model = None
        self.train_features = self.build_data(self.train_data)
        self.validation_features = self.build_data(self.validation_data)
        self.train_and_evaluate(self.train_features, self.validation_features)

    def merge_articles(self, articles: list[list[str]]):
        total_articles = list()
        for article in articles:
            total_articles += article
        return total_articles

    def get_article_sent_position(self, articles: list[list[str]]):
        article_sent_positions = list()
        for article in articles:
            pos = [idx+1 for (idx, _) in enumerate(article)]
            article_sent_positions += pos
        return article_sent_positions

    def get_article_headline(self, articles: list[list[str]], headlines: list):
        sent_headlines = list()
        for (idx, article) in enumerate(articles):
            for i in range(len(article)):
                sent_headlines.append(" ".join(headlines[idx]))
        return sent_headlines

    def merge_labels(self, labels: list[numpy.ndarray]):
        return numpy.concatenate((labels))

    def train(self, data):
        # train classifier and store model
        self.model = self.classifier.train(data)

    def test(self, data):
        # return accuracy for the model on input data
        return nltk.classify.accuracy(self.model, data)

    def train_and_evaluate(self, train, test):
        self.train(train)
        return self.test(test)

    def extract_features(self, sent: str, position: int, headline: str):
        # create a dict of features from a sent
        return {
            "length": len(sent),
            "position": position,
            "headline_similarity": self.sentence_similarity(sent, headline),
            "headline_common_word_count": self.headline_commin_words(sent, headline)
        }

    def headline_commin_words(self, sent, headline):
        sent_words = nltk.tokenize.word_tokenize(sent)
        headline_words = nltk.tokenize.word_tokenize(headline)
        return len(set(sent_words).intersection(headline_words))

    def build_data(self, data):
        # populate the features with the above function
        return [(self.extract_features(sent, pos, headline), label)
                for (sent, pos, headline, label) in data]

    def summarize(self, sents: list, headline: str, num_of_sent: int = 5, language="german"):
        generated_summary_sents = list()
        for (idx, sent) in enumerate(sents):
            feature = self.extract_features(sent, idx+1, headline)
            predicted_label = self.model.classify(feature)
            if predicted_label > 0:
                # sentence belongs to summary
                generated_summary_sents.append(sent)
        self.summary_sents = generated_summary_sents
        self.model.show_most_informative_features(20)
        return generated_summary_sents
