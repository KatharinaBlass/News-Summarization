from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import numpy as np
import networkx as nx  # version 2.6 !
import re


class TextRank:
    def __init__(self):
        self.language = "german"
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.summary_sents = None

    def filter_characters(self, sent: str):
        # regex removes punctuation, []-brackets (but keeps its content), () (without keeping its content) and other special symbols related to speech
        return re.sub("\[(.*)\]|(\(.*\))|([\.\,\!\?]+)|([\'\`\"\-\_\:\;\n]+)", "\g<1>", sent)

    def filter_POS_tags(self, sent_tokens: list[str]):
        tags = ["NN", "NNS", "NNP",  "JJ", "JJR", "JJS"]
        return [word for (word, tag) in pos_tag(
            sent_tokens) if tag in tags]

    def remove_stopwords(self, sent):
        # TODO: think about adding more news specific stopwords
        sw = stopwords.words(self.language)
        return [w for w in sent if w not in sw]

    def stemming(self, sent):
        return [self.stemmer.stem(w) for w in sent]

    def lemmatizing(self, sent: str):
        return [self.lemmatizer.lemmatize(w) for w in sent]

    def clean_sentences(self, sents: list[str]):
        filtered_sents = []
        for sent in sents:
            s = sent.lower()
            s = self.filter_characters(s)
            s = word_tokenize(s)
            s = self.remove_stopwords(s)
            #s = self.stemming(s)
            s = self.lemmatizing(s)
            #s = self.filter_POS_tags(s)
            s = " ".join(s).strip()
            filtered_sents.append(s)

        return filtered_sents

    def build_similarity_matrix(self, sents: list):
        matrix = np.zeros((len(sents), len(sents)))

        for i in range(len(sents)):
            for j in range(len(sents)):
                # same sentences --> no need to add in matrix
                if i == j:
                    continue
                matrix[i][j] = self.sentence_similarity(sents[i], sents[j])
        return matrix

    def sentence_similarity(self, sent1: str, sent2: str):
        # 1 means sentences are maximal similar, -1 means the opposite

        # if one sent is empty, it will cause an error in the cosine distance calculation --> return -1 (means it is unsimilar --> sentence will not be chosen for summary)
        if len(sent1) == 0 or len(sent2) == 0:
            return -1

        lexicon = list(set(sent1 + sent2))

        vector1 = self.build_vector(lexicon, sent1)
        vector2 = self.build_vector(lexicon, sent2)

        return 1 - cosine_distance(vector1, vector2)

    def build_vector(self, lexicon: list, sent: str):
        vector = np.zeros(len(lexicon))

        for w in sent:
            vector[lexicon.index(w)] += 1

        return vector

    def get_top_n_sentences(self, ranks: dict[any, float], sents: list, top_n: int):
        ranked_sentences = sorted(
            ((ranks[i], s) for i, s in enumerate(sents)), reverse=True)

        return [sent for (_, sent) in ranked_sentences[:top_n]]

    def apply_pagerank(self, similarity_matrix: np.ndarray):
        similarity_graph = nx.from_numpy_array(
            similarity_matrix)

        return nx.pagerank_numpy(similarity_graph)

    def summarize(self, sents: list, num_of_sent: int = 5, language="german"):
        self.language = language
        cleaned_sentences = self.clean_sentences(sents)
        sim_matrix = self.build_similarity_matrix(cleaned_sentences)
        ranks = self.apply_pagerank(sim_matrix)
        res = self.get_top_n_sentences(ranks, sents, num_of_sent)
        self.summary_sents = res
        return res
