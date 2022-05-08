from NB_classifier import NaiveBayesSummarizer
from basic_summarizer import BasicSummarizer
from data_loader import DataLoader
from text_rank import TextRankSummarizer
from evaluator import Evaluator
from tfidf import tfidfSummarizer, tfidfScikitSummarizer
from lead_n import LeadNSummarizer
from word_probability import SumBasicSummarizer
from nltk import sent_tokenize
import numpy
import json
import os.path


class ExperimentRunner:
    def __init__(self):
        self.evaluator = Evaluator()

    def run_single(self, article_sents: list, summary: str, headline: str, summarizer: BasicSummarizer, num_sents: int = 2, with_print=False):
        # number of sentences in summary makes a huge difference in the rouge evaluation --> best score with 2-sents summary
        generated_summary_sents = summarizer.summarize(
            article_sents, headline=headline, num_of_sent=num_sents)
        generated_summary = " ".join(generated_summary_sents)
        rouge_scores = self.evaluator.rouge_score_single(
            summary, generated_summary)

        if with_print:
            # print("original article: ", " ".join(article_sents))
            # print(" ")
            print("original summary: ", summary)
            print(" ")
            print("generated summary: ", generated_summary)
            print(" ")
            print("rouge scores: ")
            self.evaluator.pretty_print_scores(rouge_scores)

        return rouge_scores

    def run(self, data: dict, summarizer: BasicSummarizer, num_sents: int = 2):
        articles = data["articles"]
        gold_summaries = data["summaries"]
        headlines = data["headlines"]
        rouge_scores_list = list()

        for (idx, article) in enumerate(articles):
            rouge_scores = self.run_single(
                article, " ".join(gold_summaries[idx]), " ".join(headlines[idx]), summarizer, num_sents)
            rouge_scores_list.append(rouge_scores)

        avg_rouge_scores = self.evaluator.calculate_avg_rouge_score(
            rouge_scores_list)
        print("avg scores:")
        self.evaluator.pretty_print_scores(avg_rouge_scores)
        return avg_rouge_scores

    def make_labels(self, articles: list[list[str]], abstractive_summaries: list[list[str]]):
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

        # print(generated_summary_sents, summary_score)
        return generated_summary_sents


print("data peprocessing...")

data_loader = DataLoader()
german_test_data = data_loader.get_formatted_data()
#german_train_data = data_loader.get_formatted_data("train")
#german_validation_data = data_loader.get_formatted_data("validation")

experiment = ExperimentRunner()
text_rank_summerizer = TextRankSummarizer()
tfidf_summerizer = tfidfSummarizer()
tfidf_scikit_summarizer = tfidfScikitSummarizer()
lead_n_summarizer = LeadNSummarizer()
sum_basic_summarizer = SumBasicSummarizer()


# example_article = german_test_data["articles"][0]
# example_summary = " ".join(german_test_data["summaries"][0])

example_test_data = dict()
example_test_data["articles"] = german_test_data["articles"][:1000]
example_test_data["summaries"] = german_test_data["summaries"][:1000]
example_test_data["headlines"] = german_test_data["headlines"][:1000]


# experiment.greedy_convert_labels(example_summary, example_article)
"""
labels = dict()
if os.path.exists('extractive_labels.json'):
    with open('extractive_labels.json') as json_file:
        labels = json.load(json_file)

else:
    train_labels = experiment.make_labels(
        german_train_data["articles"], german_train_data["summaries"])
    validation_labels = experiment.make_labels(
        german_validation_data["articles"], german_validation_data["summaries"])
    labels = {
        'train': train_labels,
        'validation': validation_labels
    }
    json_string = json.dumps(labels)
    with open('extractive_labels.json', 'w') as outfile:
        outfile.write(json_string)

print("model training...")


nb_summarizer = NaiveBayesSummarizer(
    german_train_data["articles"][:10000], labels["train"][:10000], german_train_data["headlines"][:10000], german_validation_data["articles"][:10], labels["validation"][:10], german_validation_data["headlines"][:10])

print("summarizing...")

avg_test_scores = experiment.run(
    example_test_data, nb_summarizer)

"""
"""
rouge_scores_list = list()
for (idx, abstractive_summary) in enumerate(example_test_data["summaries"]):
    abstractive_summary = " ".join(abstractive_summary)
    article = example_test_data["articles"][idx]
    extractive_summary = " ".join(experiment.greedy_convert_labels(
        abstractive_summary, article))
    rouge_scores = experiment.evaluator.rouge_score_single(
        abstractive_summary, extractive_summary)

    rouge_scores_list.append(rouge_scores)

avg_rouge_scores = experiment.evaluator.calculate_avg_rouge_score(
    rouge_scores_list)
print("avg scores:")
experiment.evaluator.pretty_print_scores(avg_rouge_scores)
"""


# scores = experiment.run_single(example_article, example_summary, lead_n_summarizer, with_print=True)
#experiment = ExperimentRunner()
#sum_basic_summarizer = SumBasicSummarizer()

# tfidf_scikit_summarizer = tfidfScikitSummarizer()

print("##### sum basic")
avg_test_scores = experiment.run(example_test_data, sum_basic_summarizer)
