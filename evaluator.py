from rouge_score import rouge_scorer, scoring
import pandas as pd


class Evaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    def rouge_score_single(self, reference_summary: str, produced_summary: str):
        return self.rouge_scorer.score(reference_summary, produced_summary)

    def add_scores(self, score_dict: dict, score_to_add: scoring.Score):
        score_dict["precision"] += score_to_add.precision
        score_dict["recall"] += score_to_add.recall
        score_dict["fmeasure"] += score_to_add.fmeasure
        return score_dict

    def calculate_avg_rouge_score(self, rouge_scores_list: list):
        num_scores_in_list = len(rouge_scores_list)
        summed_scores = dict()
        avg_scores = dict()

        for rouge_score in rouge_scores_list:
            for k in rouge_score.keys():
                if k not in summed_scores.keys():
                    summed_scores[k] = {
                        "precision": 0,
                        "recall": 0,
                        "fmeasure": 0
                    }
                summed_scores[k] = self.add_scores(
                    summed_scores[k], rouge_score[k])

        for k in summed_scores.keys():
            avg_p = summed_scores[k]["precision"] / num_scores_in_list
            avg_r = summed_scores[k]["recall"] / num_scores_in_list
            avg_f = summed_scores[k]["fmeasure"] / num_scores_in_list
            score = scoring.Score(avg_p, avg_r, avg_f)
            avg_scores[k] = score

        return avg_scores

    def pretty_print_scores(self, scores: dict[str, scoring.Score]):
        table_rows = list()
        table_index = scores.keys()
        for score in scores.values():
            new_row = [score.precision, score.recall, score.fmeasure]
            table_rows.append(new_row)

        df = pd.DataFrame(
            table_rows, columns=['precision', 'recall', 'f-measure'], index=table_index)
        print(df)
