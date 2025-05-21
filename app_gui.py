import tkinter as tk
from tkinter import messagebox
import joblib
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Metin temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Model ve vektörleştirici yükle
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Tahmin fonksiyonu
def predict_sentiment():
    user_input = entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Uyarı", "Lütfen bir yorum girin.")
        return
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result_label.config(text=f"Tahmin: {prediction}", fg="green" if prediction == "positive" else "red")

# Arayüz oluştur
root = tk.Tk()
root.title("IMDB Duygu Analizi")

tk.Label(root, text="Yorumunuzu girin:").pack(pady=10)

entry = tk.Text(root, height=5, width=50)
entry.pack(pady=5)

predict_button = tk.Button(root, text="Tahmin Et", command=predict_sentiment)
predict_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
