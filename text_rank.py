from nltk import word_tokenize
import numpy as np
import networkx as nx  # version 2.6 !
from basic_summarizer import BasicSummarizer


class TextRankSummarizer(BasicSummarizer):
    def build_similarity_matrix(self, sents: list):
        matrix = np.zeros((len(sents), len(sents)))

        for i in range(len(sents)):
            for j in range(len(sents)):
                # same sentences --> no need to add in matrix
                if i == j:
                    continue
                matrix[i][j] = self.sentence_similarity(sents[i], sents[j])
        return matrix

    def build_vector(self, lexicon: list, sent: str):
        vector = np.zeros(len(lexicon))

        for w in sent:
            vector[lexicon.index(w)] += 1

        return vector

    def apply_pagerank(self, similarity_matrix: np.ndarray):
        similarity_graph = nx.from_numpy_array(
            similarity_matrix)

        return nx.pagerank_numpy(similarity_graph)

    def clean_sentences(self, sents: list[str]):
        filtered_sents = []
        for sent in sents:
            s = sent.lower()
            s = self.filter_characters(s)
            s = word_tokenize(s)
            s = self.remove_stopwords(s)
            # s = self.stemming(s)
            s = self.lemmatizing(s)
            # s = self.filter_POS_tags(s)
            s = " ".join(s).strip()
            filtered_sents.append(s)

        return filtered_sents

    def summarize(self, sents: list, num_of_sent: int = 5, language="german"):
        self.language = language
        cleaned_sentences = self.clean_sentences(sents)
        sim_matrix = self.build_similarity_matrix(cleaned_sentences)
        ranks = self.apply_pagerank(sim_matrix)
        res = self.get_top_n_sentences(ranks, sents, num_of_sent)
        self.summary_sents = res
        return res
