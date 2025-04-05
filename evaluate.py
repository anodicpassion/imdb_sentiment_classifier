from data.load_data import load_imdb_data
from tensorflow.keras.models import load_model

# Load model and data
model = load_model("saved_model/model.h5")
_, _, x_test, y_test = load_imdb_data()

# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {acc:.4f}")
