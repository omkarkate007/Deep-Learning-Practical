# Practical 4: Anomaly Detection using Autoencoder

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

# Generate normal data and artificial anomalies
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 10)).astype("float32")
anomaly_data = np.random.normal(loc=5, scale=1, size=(100, 10)).astype("float32")

# Split normal data; train only on normal observations
x_train, x_test_normal = train_test_split(
    normal_data, test_size=0.2, random_state=42
)

# Encoder
input_layer = Input(shape=(10,))
encoded = Dense(8, activation="relu")(input_layer)
latent = Dense(4, activation="relu")(encoded)

# Decoder
decoded = Dense(8, activation="relu")(latent)
output_layer = Dense(10, activation="linear")(decoded)

# Build and compile autoencoder
autoencoder = Model(input_layer, output_layer)
autoencoder.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Train the autoencoder to reconstruct normal data
autoencoder.fit(
    x_train, x_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Calculate training reconstruction errors and set threshold
train_reconstructed = autoencoder.predict(x_train, verbose=0)
train_errors = np.mean(np.square(x_train - train_reconstructed), axis=1)
threshold = np.percentile(train_errors, 95)

# Test on normal and anomalous observations
test_data = np.concatenate([x_test_normal, anomaly_data], axis=0)
reconstructed = autoencoder.predict(test_data, verbose=0)
errors = np.mean(np.square(test_data - reconstructed), axis=1)

# Error above threshold is flagged as a possible anomaly
predictions = errors > threshold

print("Reconstruction Error Threshold:", threshold)
print("Total Test Observations:", len(test_data))
print("Detected Anomalies:", np.sum(predictions))
print("Actual Artificial Anomalies:", len(anomaly_data))

plt.hist(errors, bins=30)
plt.axvline(threshold, color="red", label="Threshold")
plt.title("Autoencoder Anomaly Detection")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.legend()
plt.show()
