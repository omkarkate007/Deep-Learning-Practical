# Practical 6: Transfer Learning using MobileNetV2 on CIFAR-10
# Note: This is image classification using transfer learning, not full
# object detection with bounding boxes.

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Resize images to 96 x 96
x_train = tf.image.resize(x_train, (96, 96)).numpy()
x_test = tf.image.resize(x_test, (96, 96)).numpy()

# MobileNetV2 preprocessing
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# Load pre-trained ImageNet model without its original top classifier
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(96, 96, 3)
)

# Freeze pre-trained convolutional base
base_model.trainable = False

# Add a classifier for 10 CIFAR-10 classes
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the new classifier
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predict one image
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]
prediction = model.predict(x_test[:1])
predicted_class = int(prediction.argmax())
print("Predicted Class:", class_names[predicted_class])
print("Actual Class:", class_names[int(y_test[0][0])])

# Display original image (undo MobileNetV2 preprocessing)
display_image = (x_test[0] + 1.0) / 2.0
plt.imshow(display_image.clip(0, 1))
plt.title("Predicted: " + class_names[predicted_class])
plt.axis("off")
plt.show()

# Optional fine-tuning example (run after initial training if desired):
# Unfreeze a small number of final layers and recompile with a low learning rate.
# base_model.trainable = True
# for layer in base_model.layers[:-20]:
#     layer.trainable = False
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )
# model.fit(x_train, y_train, epochs=2, batch_size=64,
#           validation_split=0.1)
