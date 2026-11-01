# Practical 5: Continuous Bag of Words (CBOW) Model

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense
from tensorflow.keras.preprocessing.text import Tokenizer

# Small example text corpus
sentences = [
    "I love deep learning",
    "I love machine learning",
    "deep learning is powerful",
    "machine learning is useful",
    "I enjoy learning Python"
]

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

print("Vocabulary:", word_index)

# Generate context-target pairs using one word on each side
X = []
Y = []
window = 1

for sentence in sequences:
    for i in range(window, len(sentence) - window):
        context = sentence[i-window:i] + sentence[i+1:i+window+1]
        X.append(context)
        Y.append(sentence[i])

X = np.array(X, dtype="int32")
Y = np.array(Y, dtype="int32")

print("Training examples:", len(X))

# Define CBOW model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=10, input_length=2),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(X, Y, epochs=100, verbose=0)
print("CBOW model training completed.")

# Predict target word for first context pair
prediction = model.predict(X[:1], verbose=0)
predicted_id = int(np.argmax(prediction[0]))

print("Context word IDs:", X[0])
print("Predicted Word:", tokenizer.index_word.get(predicted_id, "Unknown"))
print("Actual Target Word:", tokenizer.index_word[Y[0]])
