from typing import Dict
from data_loader import DataLoader
from text_rank import TextRank
from evaluator import Evaluator


class ExperimentRunner:
    def __init__(self):
        self.evaluator = Evaluator()

    def run_single(self, article_sents: list, summary: str, summarizer: TextRank, num_sents: int = 2, with_print=False):
        # number of sentences in summary makes a huge difference in the rouge evaluation --> example article idx=2 is better with 2-sents summary than with 4-sents summary
        generated_summary_sents = summarizer.summarize(
            article_sents, num_sents)
        generated_summary = " ".join(generated_summary_sents)
        rouge_scores = self.evaluator.rouge_score_single(
            summary, generated_summary)

        if with_print:
            print("original article: ", " ".join(article_sents))
            print(" ")
            print("original summary: ", summary)
            print(" ")
            print("generated summary: ", generated_summary)
            print(" ")
            print("rouge scores: ")
            self.evaluator.pretty_print_scores(rouge_scores)

        return rouge_scores

    def run(self, data: dict, summarizer: TextRank, num_sents: int = 2):
        articles = data["articles"]
        gold_summaries = data["summaries"]
        rouge_scores_list = list()

        for (idx, article) in enumerate(articles):
            rouge_scores = self.run_single(
                article, " ".join(gold_summaries[idx]), summarizer, num_sents)
            rouge_scores_list.append(rouge_scores)

        avg_rouge_scores = self.evaluator.calculate_avg_rouge_score(
            rouge_scores_list)
        print("avg scores:")
        self.evaluator.pretty_print_scores(avg_rouge_scores)
        return avg_rouge_scores


data_loader = DataLoader()
german_test_data = data_loader.get_formatted_data()
experiment = ExperimentRunner()
text_rank_summerizer = TextRank()

"""
example_article = german_test_data["articles"][0]
example_summary = " ".join(german_test_data["summaries"][0])

scores = experiment.run_single(
    example_article, example_summary, text_rank_summerizer, with_print=True)
"""
example_test_data = dict()
example_test_data["articles"] = german_test_data["articles"][:100]
example_test_data["summaries"] = german_test_data["summaries"][:100]

avg_test_scores = experiment.run(example_test_data, text_rank_summerizer)