from nltk.tokenize import RegexpTokenizer
import numpy as np
import re
from helpers import Helper


class BasicSummarizer:
    def __init__(self, language: str):
        self.language = language
        self.helper = Helper(self.language)

    def filter_parantheses(self, sent: str):
        # regex removes []-brackets (but keeps its content) and () (without keeping its content)
        return re.sub("\[(.*)\]|(\(.*\))", "\g<1>", sent)

    def remove_stopwords(self, sent):
        sw = self.helper.get_stopwords()
        return [w for w in sent if w not in sw]

    def remove_punctuation(self, sent):
        punct = [".", ",", ";", ":", "-", "_", "!", "?"]
        return [w for w in sent if w not in punct]

    def stemming(self, sent):
        return [self.helper.stem(w) for w in sent]

    def lemmatizing(self, sent):
        return self.helper.lemmatize(" ".join(sent))

    def clean_sentences(self, sents: list[str]):
        filtered_sents = []
        for sent in sents:
            cleaned_sent = self.clean_sent(sent)
            filtered_sents.append(cleaned_sent)

        return filtered_sents

    def clean_sent(self, sent: str):
        s = sent.lower()
        s = self.filter_parantheses(s)
        #tokenizer = RegexpTokenizer(r'\w+')
        #s = tokenizer.tokenize(s)
        s = self.helper.tokenize_words(s)
        s = self.remove_punctuation(s)
        s = self.remove_stopwords(s)
        s = self.lemmatizing(s)
        #s = self.stemming(s)
        s = " ".join(s).strip()
        return s

    def get_top_n_sentences(self, scores, sents: list, top_n: int):
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sents)), reverse=True)

        return [sent for (_, sent) in ranked_sentences[:top_n]]

    def summarize(self):
        raise NotImplementedError(
            "This method should be overriden in subclasses")
