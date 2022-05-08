from basic_summarizer import BasicSummarizer
from nltk import word_tokenize, FreqDist, RegexpTokenizer


class SumBasicSummarizer(BasicSummarizer):
    def score_sentences(self, sents: list[str], word_probs: dict[str, float]):
        sentences_scores = list()
        for (idx, sent) in enumerate(sents):
            sent_weight = 0
            words = word_tokenize(sent)
            if len(words) > 0:
                for word in words:
                    if word in word_probs.keys():
                        sent_weight += word_probs[word]
                sent_weight /= len(words)
            sentences_scores.append((sent_weight, sent, idx))
        return sorted(sentences_scores, reverse=True)

    def get_word_probabilities(self, article: str):
        article_words = word_tokenize(article)
        num_words = len(article_words)
        word_freq = FreqDist(article_words).most_common()
        word_probs = dict()
        for (word, count) in word_freq:
            word_probs[word] = count / num_words
        return word_probs

    def select_sentence(self, ranked_sentences: list[tuple[float, str, int]], most_important_word: str, already_selected_sent_idx: list[int]):
        for (_, sent, idx) in ranked_sentences:
            if idx in already_selected_sent_idx:
                continue
            words = word_tokenize(sent)
            if most_important_word in words:
                return idx

    def get_most_important_word(self, word_probs: dict[str, float]):
        return max(word_probs, key=word_probs.get)

    def update_word_probabilities(self, word_probs: dict[str, float], words_to_update: list[str]):
        for word in words_to_update:
            old_word_prob = word_probs[word]
            word_probs[word] = old_word_prob * old_word_prob
        return word_probs

    def select_n_sentences(self, word_probs: dict[str, float], n: int, sents: list[str]):
        selected_sentences_indexes = list()
        while len(selected_sentences_indexes) < n:
            ranked_sentences = self.score_sentences(sents, word_probs)
            most_important_word = self.get_most_important_word(word_probs)
            new_sent_idx = self.select_sentence(
                ranked_sentences, most_important_word, selected_sentences_indexes)
            selected_sentences_indexes.append(new_sent_idx)
            words_to_update = set(word_tokenize(sents[new_sent_idx]))
            word_probs = self.update_word_probabilities(
                word_probs, words_to_update)
        return selected_sentences_indexes

    def clean_sent(self, sent: str):
        s = sent.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        s = tokenizer.tokenize(s)
        s = self.remove_stopwords(s)
        s = self.lemmatizing(s)
        s = " ".join(s).strip()
        return s

    def summarize(self, sents: list[str], headline: str = None, num_of_sent: int = 5, language="german"):
        self.language = language
        cleaned_sentences = self.clean_sentences(sents)
        article = " ".join(cleaned_sentences)
        word_probs = self.get_word_probabilities(article)
        selected_indexes = self.select_n_sentences(
            word_probs, num_of_sent, cleaned_sentences)
        return [sents[idx] for idx in selected_indexes]
