from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx  # version 2.6 !


class TextRank:
    def __init__(self, text):
        self.text = text

    def clean_sentences(self, sents):
        # TODO: think about adding more news specific stopwords & maybe filter pos tags (nouns, adjectives and verbs are probably most relevant), maybe also stem or lemmatize words
        englisch_stopwords = stopwords.words("english")
        lower_sents = [s.lower() for s in sents]
        filtered_sents = [
            s for s in lower_sents if s not in englisch_stopwords]
        return filtered_sents

    def build_similarity_matrix(self, sents):
        matrix = np.zeros((len(sents), len(sents)))

        for i in range(len(sents)):
            for j in range(len(sents)):
                # same sentences --> no need to add in matrix
                if i == j:
                    continue
                matrix[i][j] = self.sentence_similarity(sents[i], sents[j])
        return matrix

    def sentence_similarity(self, sent1, sent2):
        # 1 means sentences are maximal similar, -1 means the opposite
        lexicon = list(set(sent1 + sent2))

        vector1 = self.build_vector(lexicon, sent1)
        vector2 = self.build_vector(lexicon, sent2)

        return 1 - cosine_distance(vector1, vector2)

    def build_vector(self, features, sent):
        vector = np.zeros(len(features))

        for w in sent:
            vector[features.index(w)] += 1

        return vector

    def get_top_n_sentences(self, ranks, sents, top_n):
        ranked_sentences = sorted(
            ((ranks[i], s) for i, s in enumerate(sents)), reverse=True)

        return [sent for (_, sent) in ranked_sentences[:top_n]]

    def apply_pagerank(self, similarity_matrix):
        similarity_graph = nx.from_numpy_array(
            similarity_matrix)

        return nx.pagerank(similarity_graph)

    def summarize(self, number_of_sent=5):
        cleaned_sentences = self.clean_sentences(self.text)
        sim_matrix = self.build_similarity_matrix(cleaned_sentences)
        ranks = self.apply_pagerank(sim_matrix)
        return self.get_top_n_sentences(ranks, self.text, number_of_sent)
