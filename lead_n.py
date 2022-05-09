from basic_summarizer import BasicSummarizer


class LeadNSummarizer(BasicSummarizer):
    def __init__(self, language: str):
        super().__init__(language)

    def rank_sentences(self, sents: list[str]):
        max_score = len(sents)
        return [max_score-idx for (idx, _) in enumerate(sents)]

    def summarize(self, sents: list, headline: str = None, num_of_sent: int = 5):
        ranks = self.rank_sentences(sents)
        return self.get_top_n_sentences(ranks, sents, num_of_sent)
