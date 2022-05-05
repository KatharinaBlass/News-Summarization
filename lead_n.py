from basic_summarizer import BasicSummarizer


class LeadNSummarizer(BasicSummarizer):
    def rank_sentences(self, sents: list[str]):
        max_score = len(sents)
        return [max_score-idx for (idx, _) in enumerate(sents)]

    def summarize(self, sents: list, num_of_sent: int = 5, language="german"):
        self.language = language
        ranks = self.rank_sentences(sents)
        res = self.get_top_n_sentences(ranks, sents, num_of_sent)
        self.summary_sents = res
        return res
