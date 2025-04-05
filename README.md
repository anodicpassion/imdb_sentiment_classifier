# 🎬 IMDB Sentiment Classifier (RNN & LSTM with TensorFlow)

This project is a deep learning-based sentiment classifier trained on the IMDB movie review dataset. It uses Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) layers to classify movie reviews as **positive** or **negative**.

---

## 📁 Project Structure

```
.
├── LICENSE
├── README.md
├── data
│    ├── __pycache__
│    │    └── load_data.cpython-312.pyc
│    └── load_data.py
├── evaluate.py
├── file_structure
├── models
│    ├── __pycache__
│    │   └── lstm_model.cpython-312.pyc
│    └── lstm_model.py
├── requirements.txt
├── test.py
└── train.py

5 directories, 11 files
```
---

## 🚀 Features

- Bidirectional LSTM network
- Word embedding layer
- Dropout and recurrent dropout for regularization
- Batch normalization
- Early stopping to prevent overfitting
- Modular and reusable code structure
- CLI-based training, testing, and evaluation

---

## 🧠 Model Overview

The model architecture includes:
- **Embedding Layer**
- **Bidirectional LSTM (2 layers)**
- **Batch Normalization**
- **Dense + Dropout**
- **Sigmoid Activation** for binary classification

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/imdb_sentiment_classifier.git
cd imdb_sentiment_classifier
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Installing Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Running the Project

### 1. Train the model
```bash
python3 train.py
```

### 2. Evaluate
```bash
python3 evaluate.py
```

### 3. Test with custom input
```bash
python3 test.py
```

## 📊 Accuracy

With the current setup, the model achieves 88–91% test accuracy depending on randomness and review complexity.

## 📜 License
This project is licensed under the GNU General Public License v3.0.