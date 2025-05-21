import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Metin temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# 1. Veri Yükle
df = pd.read_csv("imdb_dataset.csv")  # Veriyi buraya yükle

# 2. Veriyi temizle
df["review"] = df["review"].apply(clean_text)

# 3. Vektörleştir
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# 4. Model Eğit
model = MultinomialNB()
model.fit(X, y)

# 5. Model ve vektörleştiriciyi kaydet
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("Model ve vektörleştirici kaydedildi.")
