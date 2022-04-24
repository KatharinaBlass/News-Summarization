from data_loader import DataLoader
from text_rank import TextRank

d = DataLoader()
article = d.test_data["articles"][1]
t = TextRank(article)
print("original article: ", " ".join(article))
print(" ")
res = t.summarize(4)
print("summary: ", " ".join(res))
