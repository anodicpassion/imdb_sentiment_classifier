from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load word index
word_index = imdb.get_word_index()
index_word = {v + 3: k for k, v in word_index.items()}
index_word[0] = "<PAD>"

# Sample review (numerical form)
sample_review = "This movie was amazing and thrilling with excellent performances"
words = sample_review.lower().split()
encoded = [word_index.get(word, 2) for word in words]  # 2 is "unknown"

# Pad and predict
padded = pad_sequences([encoded], maxlen=300)
model = load_model("saved_model/model.h5")
prediction = model.predict(padded)

print(f"\n🔍 Sentiment Prediction: {'Positive' if prediction[0][0] > 0.5 else 'Negative'} (Confidence: {prediction[0][0]:.4f})")
