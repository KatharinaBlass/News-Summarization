from basic_summarizer import BasicSummarizer
import numpy
import nltk
from nltk.classify import NaiveBayesClassifier


class NaiveBayesSummarizer(BasicSummarizer):
    def __init__(self, train_articles: list[list[str]], train_labels: list[numpy.ndarray], train_headlines: list, validate_articles: list[list[str]], validate_labels: list[numpy.ndarray], validate_headlines: list, language: str):
        super().__init__(language)
        self.indicator_words = {
            "en": [
                "therefore", "thus", "consequently", "this proves", "as a result", "this suggests that", "in summary", "in conclusion", "in a nutshell", "in a word", "to conclude"
            ],
            "de": [
                "daher", "folglich", "deshalb", "darum", "demnach", "deswegen", "dies beweist", "als ergebnis", "infolgedessen", "daraufhin", "dies legt nahe", "zusammenfassend", "abschließend", "in einem wort", "abschließend"
            ],
            "fr": [
                "donc", "ainsi", "par conséquent", "en conséquence", "cela prouve", "comme résultat", "ceci suggère", "en résumé", "pour résumer", "en bref", "en conclusion", "finalement"
            ],
            "es": [
                "por lo tanto", "así", "en consecuencia", "esto demuestra", "como resultado", "esto sugiere que", "en resumen", "en conclusión", "en pocas palabras", "para concluir"
            ],
            "ru": [
                "поэтому", "таким образом", "следовательно", "в результате", "в заключение", "в двух словах", "в заключение"
            ],
            "tu": [
                "bu nedenle", "böylece", "sonuç olarak", "bu kanıtlıyor", "sonuç olarak", "bu şunu gösteriyor", "özetle", "sonuç olarak", "özetle", "tek kelimeyle", "sonlandırmak"
            ]
        }

        self.classifier = NaiveBayesClassifier
        self.model = None

        self.train_data = self.format_data(
            train_articles, train_labels)
        self.validation_data = self.format_data(
            validate_articles, validate_labels)

        self.train_features = self.build_data(self.train_data, train_headlines)
        self.validation_features = self.build_data(
            self.validation_data, validate_headlines)
        self.train_and_evaluate(self.train_features, self.validation_features)

    def format_data(self, articles: list, label: list):
        (sentences, sentence_article_indexes,
         sentence_positions, sentence_topic_words) = self.extract_flat_sentence_attributes(articles)
        sentence_labels = self.merge_labels(label)

        return list(zip(sentences, sentence_positions, sentence_labels, sentence_topic_words, sentence_article_indexes))

    def extract_flat_sentence_attributes(self, articles: list[list[str]]):
        flat_articles_sents = list()
        flat_articles_indexes = list()
        flat_articles_sent_positions = list()
        flat_articles_topic_words = list()
        for (idx, article) in enumerate(articles):
            flat_articles_sents += article
            topic_words = self.get_most_important_words(article)
            for i in range(len(article)):
                flat_articles_indexes.append(idx)
                flat_articles_sent_positions.append(i)
                flat_articles_topic_words.append(topic_words)
        return (flat_articles_sents, flat_articles_indexes, flat_articles_sent_positions, flat_articles_topic_words)

    def merge_labels(self, labels: list[numpy.ndarray]):
        return numpy.concatenate(labels)

    def get_most_important_words(self, sents: list):
        cleaned_sents = self.clean_sentences(sents)
        words = self.helper.tokenize_words(" ".join(cleaned_sents))
        important_words_freq = nltk.FreqDist(words).most_common()
        important_words = [word for (word, freq)
                           in important_words_freq if int(freq) > 1]
        return important_words

    def train(self, data):
        self.model = self.classifier.train(data)

    def test(self, data):
        print("accuracy:", nltk.classify.accuracy(self.model, data))

    def train_and_evaluate(self, train, test):
        self.train(train)
        self.test(test)

    def extract_features(self, sent: str, position: int, topic_words: list, headline: str = None):
        cleaned_sent = self.clean_sent(sent)
        sent_words = self.helper.tokenize_words(cleaned_sent)

        features = {
            "length": len(sent_words),
            "position": position,
            "topic_words_count": self.count_common_words(sent_words, topic_words),
            "contains_indicator_words": self.contains_indicator_words(sent_words),
            "contains_upper_case_words": self.contains_upper_case_words(sent)
        }

        if headline:
            # no headlines given for english data
            cleaned_headline = self.clean_sent(headline)
            headline_words = self.helper.tokenize_words(cleaned_headline)
            features["headline_common_words_count"] = self.count_common_words(
                sent_words, headline_words)

        return features

    def contains_upper_case_words(self, sent: str):
        sent_words = self.helper.tokenize_words(sent)
        # check for upper case word, but ignore first word, since sentence beginnings are always upper case
        for word in sent_words[1:]:
            if word[0].isupper():
                return True
        return False

    def contains_indicator_words(self, sent_words: list):
        sentence = " ".join([word.lower() for word in sent_words])

        for indicator in self.indicator_words[self.language]:
            if indicator in sentence:
                return True
        return False

    def count_common_words(self, words1, words2):
        return len(set(words1).intersection(words2))

    def build_data(self, data, headlines: list[str] = None):
        return [(self.extract_features(sent, pos, topic_words, headlines[article_idx] if headlines else None), label)
                for (sent, pos, label, topic_words, article_idx) in data]

    def summarize(self, sents: list, headline: str = None, num_of_sent: int = 5):
        generated_summary_sents = list()
        topic_words = self.get_most_important_words(sents)
        for (idx, sent) in enumerate(sents):
            feature = self.extract_features(sent, idx+1, topic_words, headline)
            predicted_label = self.model.classify(feature)
            if predicted_label > 0:
                # sentence belongs to summary
                generated_summary_sents.append(sent)
        # self.model.show_most_informative_features(10)
        return generated_summary_sents
