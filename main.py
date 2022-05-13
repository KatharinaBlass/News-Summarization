from NB_classifier import NaiveBayesSummarizer
from basic_summarizer import BasicSummarizer
from data_loader import DataLoader
from text_rank import TextRankSummarizer
from evaluator import Evaluator
from tfidf import tfidfSummarizer, tfidfScikitSummarizer
from lead_n import LeadNSummarizer
from word_probability import SumBasicSummarizer
import numpy
import json
import os.path
import sys
import getopt
import time

summarizer_dict = {
    "leadn": LeadNSummarizer,
    "tfidf": tfidfScikitSummarizer,
    "textrank": TextRankSummarizer,
    "sumbasic": SumBasicSummarizer,
    "nb": NaiveBayesSummarizer
}

language_dict = {
    "de": 'german',
    "fr": 'french',
    "es": 'spanish',
    "ru": 'russian',
    "tu": 'turkish',
    "en": 'english'
}

LABEL_FILE_NAME = '_extractive_labels.json'


class ExperimentRunner:
    def __init__(self, language, algorithm, train_set_size=15000):
        self.language = language
        self.algorithm = algorithm
        self.setup(train_set_size)

    def setup(self, train_set_size):
        print("preparing data...")
        self.evaluator = Evaluator()
        self.data_loader = DataLoader(self.language)
        summarizer_class = summarizer_dict[self.algorithm]
        self.test_data = self.data_loader.get_formatted_data("test")
        if self.algorithm == "nb":
            self.labels = dict()
            self.train_data = self.data_loader.get_formatted_data(
                "train", train_set_size)
            self.validation_data = self.data_loader.get_formatted_data(
                "validation")
            self.load_labels(train_set_size)
            print("model training...")
            self.summarizer = summarizer_class(
                self.train_data["articles"], self.labels["train"], self.train_data["headlines"], self.validation_data["articles"], self.labels["validation"], self.validation_data["headlines"], language=self.language)
        else:
            self.summarizer = summarizer_class(self.language)

    def load_labels(self, train_set_size):
        if os.path.exists(self.language+LABEL_FILE_NAME):
            with open(self.language+LABEL_FILE_NAME) as json_file:
                self.labels = json.load(json_file)
                if train_set_size:
                    self.labels["train"] = self.labels["train"][:train_set_size]

        else:
            train_labels = self.make_extractive_labels(
                self.train_data["articles"], self.train_data["summaries"])
            validation_labels = self.make_extractive_labels(
                self.validation_data["articles"], self.validation_data["summaries"])
            self.labels = {
                'train': train_labels,
                'validation': validation_labels
            }
            json_string = json.dumps(self.labels)
            with open(self.language+LABEL_FILE_NAME, 'w') as outfile:
                outfile.write(json_string)

    def summarize_article(self, article_sents: list, summary: str, headline: str, summarizer: BasicSummarizer, num_sents: int = 2, with_print=False):
        generated_summary_sents = summarizer.summarize(
            article_sents, headline=headline, num_of_sent=num_sents)
        generated_summary = " ".join(generated_summary_sents)
        rouge_scores = self.evaluator.rouge_score_single(
            summary, generated_summary)
        return rouge_scores

    def run(self, num_sents: int = 3):
        print("summarizing...")
        articles = self.test_data["articles"]
        gold_summaries = self.test_data["summaries"]
        headlines = self.test_data["headlines"]
        rouge_scores_list = list()

        for (idx, article) in enumerate(articles):
            rouge_scores = self.summarize_article(
                article, " ".join(gold_summaries[idx]), " ".join(headlines[idx]) if headlines else None, self.summarizer, num_sents)
            rouge_scores_list.append(rouge_scores)

        avg_rouge_scores = self.evaluator.calculate_avg_rouge_score(
            rouge_scores_list)
        print("rouge score - language", language_dict[self.language],
              "& algorithm", self.algorithm)
        self.evaluator.pretty_print_scores(avg_rouge_scores)
        return avg_rouge_scores

    def make_extractive_labels(self, articles: list[list[str]], abstractive_summaries: list[list[str]]):
        labels = list()
        for (idx, abstractive_summary) in enumerate(abstractive_summaries):
            abstractive_summary = " ".join(abstractive_summary)
            article_sents = articles[idx]
            extractive_summary_sents = self.greedy_convert_summary(
                abstractive_summary, article_sents)
            label = numpy.zeros(len(article_sents))
            for (idx, sent) in enumerate(article_sents):
                if sent in extractive_summary_sents:
                    label[idx] = 1
            labels.append(label.tolist())
        return labels

    def greedy_convert_summary(self, target_summary: str, sents: list[str]):
        generated_summary_sents = list()
        summary_score = 0

        sents_values = list()

        for sent in sents:
            value = self.evaluator.get_fmeasure_rouge1_score_single(
                target_summary, sent)
            sents_values.append((value, sent))

        sents_values = sorted(sents_values, reverse=True)

        for (value, sent) in sents_values:
            generated_summary_sents.append(sent)
            current_summary = " ".join(generated_summary_sents)
            current_summary_score = self.evaluator.get_fmeasure_rouge1_score_single(
                target_summary, current_summary)

            if current_summary_score < summary_score:
                # current sent was a bad choice --> remove it and continue with next sent
                generated_summary_sents.pop()
            else:
                summary_score = current_summary_score
        return generated_summary_sents


def main(language, algorithm):
    print(language, algorithm)
    start_time = time.time()
    experiment = ExperimentRunner(language, algorithm)
    experiment.run()
    print("execution time --- %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "l:a:")
    except getopt.GetoptError:
        print('main.py -l <language> -a <algorithm>')
        sys.exit(2)

    supported_languages = list(language_dict.keys())
    supported_algorithms = list(summarizer_dict.keys())
    language = supported_languages[0]
    algorithm = supported_algorithms[0]

    for opt, arg in opts:
        if opt == '-l':
            if arg in supported_languages:
                language = arg
            else:
                print('language not supported - supported languages are ',
                      supported_languages)
                sys.exit(2)
        elif opt == '-a':
            if arg in supported_algorithms:
                algorithm = arg
            else:
                print('algorithm not supported - supported algorithms are ',
                      supported_algorithms)
                sys.exit(2)

    main(language=language, algorithm=algorithm)
