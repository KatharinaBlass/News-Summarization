from basic_summarizer import BasicSummarizer
import numpy
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize, RegexpTokenizer


class NaiveBayesSummarizer(BasicSummarizer):
    def __init__(self, train_articles: list[list[str]], train_labels: list[numpy.ndarray], train_headlines: list, validate_articles: list[list[str]], validate_labels: list[numpy.ndarray], validate_headlines: list):
        super().__init__()
        self.indicator_words = {
            "en": ["so", "therefore", "thus", "consequently",
                   "this proves", "as a result", "this suggests that", "in summary", "in conclusion", "in a nutshell", "in a word", "to conclude"],
            "de": ["daher", "folglich", "deshalb", "darum", "demnach", "deswegen", "dies beweist",
                   "als ergebnis", "infolgedessen", "daraufhin", "dies legt nahe", "zusammenfassend", "abschließend", "in einem wort", "abschließend"]}

        self.train_data = self.format_data(
            train_articles, train_labels, train_headlines)
        self.validation_data = self.format_data(
            validate_articles, validate_labels, validate_headlines)

        self.classifier = NaiveBayesClassifier
        self.model = None
        self.train_features = self.build_data(self.train_data)
        self.validation_features = self.build_data(self.validation_data)
        self.train_and_evaluate(self.train_features, self.validation_features)

    def format_data(self, articles: list, label: list, headlines: list):
        sentences = self.merge_articles(articles)
        sentence_positions = self.get_articles_sent_positions(articles)
        sentence_headlines = self.get_articles_headlines(articles, headlines)
        sentence_labels = self.merge_labels(label)
        sentence_topic_words = self.get_articles_topic_words(articles)

        return list(zip(sentences, sentence_positions, sentence_headlines, sentence_labels, sentence_topic_words))

    def merge_articles(self, articles: list[list[str]]):
        total_articles = list()
        for article in articles:
            total_articles += article
        return total_articles

    def get_articles_sent_positions(self, articles: list[list[str]]):
        article_sent_positions = list()
        for article in articles:
            pos = [idx+1 for (idx, _) in enumerate(article)]
            article_sent_positions += pos
        return article_sent_positions

    def get_articles_headlines(self, articles: list[list[str]], headlines: list):
        sent_headlines = list()
        for (idx, article) in enumerate(articles):
            for i in range(len(article)):
                sent_headlines.append(" ".join(headlines[idx]))
        return sent_headlines

    def merge_labels(self, labels: list[numpy.ndarray]):
        return numpy.concatenate((labels))

    def get_articles_topic_words(self, articles: list):
        articles_topic_words = list()
        for article in articles:
            topic_words = self.get_most_important_words(article)
            for i in range(len(article)):
                articles_topic_words.append(topic_words)
        return articles_topic_words

    def get_most_important_words(self, sents: list):
        cleaned_sents = self.clean_sentences(sents)
        words = word_tokenize(" ".join(cleaned_sents))
        important_words_freq = nltk.FreqDist(words).most_common()
        important_words = [word for (word, freq)
                           in important_words_freq if int(freq) > 1]
        return important_words

    def clean_sent(self, sent: str):
        s = sent.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        s = tokenizer.tokenize(s)
        #s = word_tokenize(s)
        s = self.remove_punctuation(s)
        s = self.remove_stopwords(s)
        s = self.lemmatizing(s)
        #s = self.stemming(s)
        s = " ".join(s).strip()
        return s

    def train(self, data):
        # train classifier and store model
        self.model = self.classifier.train(data)

    def test(self, data):
        # return accuracy for the model on input data
        print(nltk.classify.accuracy(self.model, data))

    def train_and_evaluate(self, train, test):
        self.train(train)
        self.test(test)

    def extract_features(self, sent: str, position: int, headline: str, topic_words: list):
        cleaned_sent = self.clean_sent(sent)
        sent_words = word_tokenize(cleaned_sent)

        cleaned_headline = self.clean_sent(headline)
        headline_words = word_tokenize(cleaned_headline)

        return {
            "length": len(sent_words),
            "position": position,
            # "headline_similarity": self.sentence_similarity(sent, headline),
            "headline_common_words_count": self.count_common_words(sent_words, headline_words),
            "topic_words_count": self.count_common_words(sent_words, topic_words),
            "contains_indicator_words": self.contains_indicator_words(sent),
            "contains_upper_case_words": self.contains_upper_case_words(sent)
        }

    def similarity_to_n_sents(self, sent, n_sents):
        total_sim = 0
        for sent2 in n_sents:
            sim = self.sentence_similarity(sent, sent2)
            total_sim += sim
        return total_sim

    def contains_upper_case_words(self, sent_words: list):
        for word in sent_words:
            if word[0].isupper():
                return True

    def contains_indicator_words(self, sent_words: list):
        sentence = " ".join([word.lower() for word in sent_words])

        for indicator in self.indicator_words["de"]:
            if indicator in sentence:
                return True

    def count_common_words(self, words1, words2):
        return len(set(words1).intersection(words2))

    def build_data(self, data):
        # populate the features with the above function
        return [(self.extract_features(sent, pos, headline, topic_words), label)
                for (sent, pos, headline, label, topic_words) in data]

    def summarize(self, sents: list, headline: str, num_of_sent: int = 5, language="german"):
        self.language = language
        generated_summary_sents = list()
        topic_words = self.get_most_important_words(sents)
        for (idx, sent) in enumerate(sents):
            feature = self.extract_features(sent, idx+1, headline, topic_words)
            predicted_label = self.model.classify(feature)
            if predicted_label > 0:
                # sentence belongs to summary
                generated_summary_sents.append(sent)
        self.summary_sents = generated_summary_sents
        # self.model.show_most_informative_features(20)
        return generated_summary_sents
