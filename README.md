# NextWord AI

## Live Demo

[Try NextWord AI](https://nextwordprediction-c2tdzgyz3ske3hheanorrl.streamlit.app/)

A deep learning based **Next Word Prediction** application that predicts the most likely next word from a given text sequence.

The project uses **Natural Language Processing (NLP)** techniques and an **LSTM (Long Short-Term Memory)** neural network to learn patterns from text and generate next-word predictions.

---

## Features

- Next-word prediction using LSTM
- NLP text preprocessing
- Tokenization and vocabulary creation
- Sequence generation
- Word embedding
- Top-5 word predictions
- Prediction confidence score
- Interactive Streamlit web interface
- Professional responsive UI
- Saved trained model for real-time inference

---

## Tech Stack

### Programming Language

- Python

### Machine Learning / Deep Learning

- TensorFlow
- Keras
- LSTM
- Word Embeddings

### NLP

- Text Cleaning
- Lowercasing
- Punctuation Removal
- Number Removal
- Stop Word Removal
- Tokenization
- Vocabulary Creation
- Sequence Generation
- Padding

### Frontend

- Streamlit
- HTML
- CSS

### Libraries

- NumPy
- TensorFlow
- Streamlit
- Pickle

---

## Project Architecture

```text
Input Text
     │
     ▼
Text Preprocessing
     │
     ▼
Tokenization
     │
     ▼
Vocabulary
     │
     ▼
Integer Sequence
     │
     ▼
Padding
     │
     ▼
Embedding Layer
     │
     ▼
LSTM
     │
     ▼
Dense + Softmax
     │
     ▼
Next Word Prediction
     │
     ▼
Top-5 Suggestions
