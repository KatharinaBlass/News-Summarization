import numpy as np
import networkx as nx  # version 2.6 !
from basic_summarizer import BasicSummarizer
from nltk.cluster.util import cosine_distance


class TextRankSummarizer(BasicSummarizer):
    def __init__(self, language: str):
        super().__init__(language)

    def build_similarity_matrix(self, sents: list):
        matrix = np.zeros((len(sents), len(sents)))

        for i in range(len(sents)):
            for j in range(len(sents)):
                if i == j:
                    continue
                matrix[i][j] = self.sentence_similarity(sents[i], sents[j])
        return matrix

    def sentence_similarity(self, sent1: str, sent2: str):
        # 1 means sentences are maximal similar, -1 means unsimilar

        # lexicon = list(set(sent1 + sent2))
        # vector1 = self.build_vector(lexicon, sent1)
        # vector2 = self.build_vector(lexicon, sent2)

        # glove embeddings result in poor scores
        vector1 = self.build_glove_vector(sent1)
        vector2 = self.build_glove_vector(sent2)

        # if one vector is empty, it will cause an error in the cosine distance calculation --> return -1 (means it is unsimilar --> sentence will not be chosen for summary)
        vector1_all_zeros = not np.any(vector1)
        vector2_all_zeros = not np.any(vector2)

        if vector1_all_zeros or vector2_all_zeros:
            return -1

        return 1 - cosine_distance(vector1, vector2)

    def build_vector(self, lexicon: list, sent: str):
        vector = np.zeros(len(lexicon))

        for w in sent:
            vector[lexicon.index(w)] += 1

        return vector

    def build_glove_vector(self, sent: str):
        if len(sent) != 0:
            v = sum([self.word_embeddings.get(w, np.zeros((100,)))
                     for w in sent.split()])/(len(sent.split())+0.001)
        else:
            v = np.zeros((100,))
        return v

    def apply_pagerank(self, similarity_matrix: np.ndarray):
        similarity_graph = nx.from_numpy_array(
            similarity_matrix)

        return nx.pagerank_numpy(similarity_graph)

    def clean_sent(self, sent: str):
        s = sent.lower()
        s = self.filter_characters(s)
        s = self.helper.tokenize_words(s)
        s = self.remove_stopwords(s)
        # s = self.stemming(s)
        s = self.lemmatizing(s)
        s = " ".join(s).strip()
        return s

    def summarize(self, sents: list, headline: str = None, num_of_sent: int = 5):
        cleaned_sentences = self.clean_sentences(sents)
        sim_matrix = self.build_similarity_matrix(cleaned_sentences)
        ranks = self.apply_pagerank(sim_matrix)
        return self.get_top_n_sentences(ranks, sents, num_of_sent)
