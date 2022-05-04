from basic_summarizer import BasicSummarizer
from nltk import word_tokenize
import math


class tfidfSummarizer(BasicSummarizer):
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
        # sum up the TF frequency of every word in a sentence and devide this by total number of words in a sentence
        sentencesScores = list()

        for tfidf_table in tfidf_matrix:
            total_score_per_sentence = 0

            count_words_in_sentence = len(tfidf_table)
            for score in tfidf_table.values():
                total_score_per_sentence += score

            if count_words_in_sentence == 0:
                sentencesScores.append(0)
            else:
                sentencesScores.append(
                    total_score_per_sentence / count_words_in_sentence)

        return sentencesScores

    def summarize(self, sents: list[str], num_of_sent: int = 5, language="german"):
        self.language = language
        cleaned_sentences = self.clean_sentences(sents)
        tokenized_sents = [word_tokenize(sent) for sent in cleaned_sentences]
        num_documents_per_word = self.count_documents_per_word(tokenized_sents)
        tf_idf_matrix = self.create_tf_idf_matrix(
            tokenized_sents, num_documents_per_word)
        sentence_scores = self.score_sentences(tf_idf_matrix)
        res = self.get_top_n_sentences(sentence_scores, sents, num_of_sent)
        return res

    def clean_sentences(self, sents: list[str]):
        # TODO: test if this method can be replaced by base class cleanup method
        filtered_sents = []
        for sent in sents:
            s = sent.lower()
            # s = self.filter_characters(s)
            s = word_tokenize(s)
            s = self.remove_stopwords(s)
            s = self.stemming(s)
            # s = self.lemmatizing(s)
            # s = self.filter_POS_tags(s)
            s = " ".join(s).strip()
            filtered_sents.append(s)

        return filtered_sents
