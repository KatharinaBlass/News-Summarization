from basic_summarizer import BasicSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math


class tfidfSummarizer(BasicSummarizer):
    def __init__(self, language: str):
        super().__init__(language)

    def create_tf_idf_matrix(self, tokenized_sents: list[list[str]], num_documents_per_word: dict):
        num_documents = len(tokenized_sents)
        tf_idf_matrix = list()

        for sent in tokenized_sents:
            tf_idf_table = {}
            number_words_in_sent = len(set(sent))
            for word in sent:
                if word not in tf_idf_table:
                    # count word frequency
                    word_freq = sent.count(word)
                    # calculate term frequency
                    term_freq = word_freq / number_words_in_sent
                    # calculate inverse document frequency
                    inverse_doc_freq = math.log10(
                        num_documents / float(num_documents_per_word[word]))
                    # multipy tf and idf to calculate tf_idf
                    tf_idf = float(term_freq * inverse_doc_freq)
                    tf_idf_table[word] = tf_idf
            tf_idf_matrix.append(tf_idf_table)
        return tf_idf_matrix

    def count_documents_per_word(self, tokenized_sents: list[list[str]]):
        documents_per_word_dict = {}

        for sent in tokenized_sents:
            for word in set(sent):
                if word in documents_per_word_dict:
                    documents_per_word_dict[word] += 1
                else:
                    documents_per_word_dict[word] = 1

        return documents_per_word_dict

    def score_sentences(self, tfidf_matrix: list[dict[str, float]]):
        # sum up the TF frequency of every word in a sentence
        sentencesScores = list()

        for tfidf_table in tfidf_matrix:
            total_score_per_sentence = 0
            for score in tfidf_table.values():
                total_score_per_sentence += score
            sentencesScores.append(total_score_per_sentence)
            """
            count_words_in_sentence = len(tfidf_table)
            if count_words_in_sentence == 0:
                sentencesScores.append(0)
            else:
                sentencesScores.append(
                    total_score_per_sentence/count_words_in_sentence)
            """
        return sentencesScores

    def summarize(self, sents: list[str], headline: str = None, num_of_sent: int = 5):
        cleaned_sentences = self.clean_sentences(sents)
        tokenized_sents = [self.helper.tokenize_words(
            sent) for sent in cleaned_sentences]
        num_documents_per_word = self.count_documents_per_word(tokenized_sents)
        tf_idf_matrix = self.create_tf_idf_matrix(
            tokenized_sents, num_documents_per_word)
        sentence_scores = self.score_sentences(tf_idf_matrix)
        return self.get_top_n_sentences(sentence_scores, sents, num_of_sent)


class tfidfScikitSummarizer(BasicSummarizer):
    def __init__(self, language: str):
        super().__init__(language)

    def tokenize(self, text):
        # increases performance a bit
        text = text.lower()
        tokens = self.helper.tokenize_words(text)
        tokens = self.remove_punctuation(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatizing(tokens)
        return tokens

    def create_tf_idf_matrix(self, sents: list[str]):
        tfidfvectorizer = TfidfVectorizer(
            analyzer='word', stop_words=self.helper.get_stopwords())
        tfidf_wm = tfidfvectorizer.fit_transform(sents)
        return tfidf_wm

    def score_sentences(self, tfidf_matrix):
        sentencesScores = list()

        num_sents = tfidf_matrix.shape[0]
        for i in range(num_sents):
            sum_weights = math.fsum(tfidf_matrix[i, :].toarray()[0])
            """
            num_words_in_sent = len(
                [t for t in tfidf_matrix[i, :].toarray()[0] if t > 0])
            if num_words_in_sent == 0:
                sentencesScores.append(sum_weights)
            else:
                sentencesScores.append(
                    sum_weights/num_words_in_sent)
            """
            sentencesScores.append(sum_weights)

        return sentencesScores

    def summarize(self, sents: list[str], headline: str = None, num_of_sent: int = 5):
        cleaned_sentences = self.clean_sentences(sents)
        tf_idf_matrix = self.create_tf_idf_matrix(cleaned_sentences)
        sentence_scores = self.score_sentences(tf_idf_matrix)
        return self.get_top_n_sentences(sentence_scores, sents, num_of_sent)
