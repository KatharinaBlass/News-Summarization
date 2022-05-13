# News Summarization

## setup

The repo requires python to be installed.
Furthermore the following packages should be installed via pip:

- nltk
- spacy
- scikit-learn
- networkx
- rouge-score
- datasets

afterwards install the spacy language models:

- python -m spacy download en_core_web_sm
- python -m spacy download fr_core_news_sm
- python -m spacy download de_core_news_sm
- python -m spacy download ru_core_news_sm
- python -m spacy download es_core_news_sm

## run

run main.py by the following command:

```
python main.py -l <language> -a <algorithm>
```

options for language l: de, en, fr, es, ru, tu

options for algorithm a: leadn, tfidf, textrank, nb, sumbasic
