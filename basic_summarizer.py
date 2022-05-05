from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import numpy as np
import re


class BasicSummarizer:
    def __init__(self):
        self.language = "german"
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.word_embeddings = {}
        self.load_glove_word_embeddings()

    def load_glove_word_embeddings(self):
        f = open('./glove/glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.word_embeddings[word] = coefs
        f.close()

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

    def remove_punctuation(self, sent):
        punct = [".", ",", ";", "-", "!", "?"]
        return [w for w in sent if w not in punct]

    def stemming(self, sent):
        return [self.stemmer.stem(w) for w in sent]

    def lemmatizing(self, sent: str):
        return [self.lemmatizer.lemmatize(w) for w in sent]

    def clean_sentences(self, sents: list[str]):
        filtered_sents = []
        for sent in sents:
            s = sent.lower()
            #tokenizer = RegexpTokenizer(r'\w+')
            #s = tokenizer.tokenize(s)
            s = word_tokenize(s)
            s = self.remove_punctuation(s)
            s = self.remove_stopwords(s)
            s = self.lemmatizing(s)
            s = " ".join(s).strip()
            filtered_sents.append(s)

        return filtered_sents

    def get_top_n_sentences(self, scores, sents: list, top_n: int):
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sents)), reverse=True)

        return [sent for (_, sent) in ranked_sentences[:top_n]]

    def summarize(self):
        raise NotImplementedError(
            "This method should be overriden in subclasses")
