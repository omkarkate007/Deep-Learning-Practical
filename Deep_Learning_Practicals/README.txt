DEEP LEARNING PRACTICALS – EXTERNAL EXAM PREPARATION
===================================================

Environment
-----------
Recommended: Google Colab or Jupyter Notebook.
Python packages:
    tensorflow
    matplotlib
    numpy
    scikit-learn

In Google Colab, TensorFlow, NumPy and Matplotlib are usually preinstalled.
If needed, run:
    !pip install tensorflow matplotlib scikit-learn

PRACTICAL 1 – STUDY OF DEEP LEARNING PACKAGES
---------------------------------------------
Aim:
Study TensorFlow, Keras, Theano and PyTorch and document their features.

Theory:
Deep Learning is a subset of Machine Learning that uses multi-layered
artificial neural networks to learn patterns from data.

TensorFlow:
- Open-source deep learning framework developed by Google.
- Supports CPU/GPU computation, automatic differentiation and distributed
  training.
- TensorFlow Lite supports deployment on mobile and edge devices.

Keras:
- High-level deep learning API.
- Makes neural-network development easier.
- Provides Sequential and Functional APIs, layers, optimizers and losses.
- Commonly used with TensorFlow.

Theano:
- Python library for numerical computation and symbolic mathematical
  expressions.
- Supports automatic differentiation and GPU computation.
- Discontinued; no longer actively maintained.

PyTorch:
- Open-source framework developed by Meta.
- Supports dynamic computation graphs, GPU acceleration and automatic
  differentiation.
- Popular in research and computer vision.

Comparison:
TensorFlow – Framework – Google – Active
Keras      – High-level API – Originally François Chollet – Active
Theano     – Numerical computation library – MILA – Discontinued
PyTorch    – Framework – Meta – Active

Code:
See practical_1_packages.py.

PRACTICAL 2 – FEEDFORWARD NEURAL NETWORK
----------------------------------------
Aim:
Implement a Feedforward Neural Network using Keras and TensorFlow on MNIST.

Theory:
A Feedforward Neural Network passes information in one direction: input,
hidden layers, then output. MNIST contains 70,000 grayscale handwritten
digit images (60,000 training and 10,000 testing), each 28 x 28 pixels.

Layers:
- Flatten: converts the 28 x 28 image into 784 values.
- Dense hidden layers: learn patterns.
- Output layer: 10 neurons, one for each digit (0–9).

Algorithm:
1. Import libraries and load MNIST.
2. Normalize pixel values to [0, 1].
3. Define the network using Keras.
4. Compile with SGD and sparse categorical cross-entropy.
5. Train, evaluate and plot loss/accuracy.

Code:
See practical_2_feedforward_mnist.py.

PRACTICAL 3 – IMAGE CLASSIFICATION USING CNN
--------------------------------------------
Aim:
Build an image-classification model using a CNN and Fashion-MNIST.

Theory:
Image classification assigns an image to one of a set of predefined
categories. A CNN learns visual features using convolution and pooling.
Fashion-MNIST has 70,000 grayscale 28 x 28 images in 10 clothing categories.

Stages:
1. Load and preprocess data; normalize pixels and add a channel dimension.
2. Define Conv2D, MaxPooling2D, Flatten and Dense layers.
3. Train using training images.
4. Estimate performance using test loss and accuracy.

Code:
See practical_3_cnn_fashion_mnist.py.

PRACTICAL 4 – ANOMALY DETECTION USING AUTOENCODER
------------------------------------------------
Aim:
Use an Autoencoder to detect unusual observations.

Theory:
An Autoencoder learns to reconstruct its input. The encoder compresses the
input into a latent representation; the decoder reconstructs the input.
It is trained using normal data. A large reconstruction error may indicate
an anomaly.

Components:
- Encoder: maps input to a smaller latent representation.
- Latent space: compressed representation.
- Decoder: reconstructs the original input.
- Reconstruction error: mean squared difference between input and output.
- Threshold: error above this value is flagged as a possible anomaly.

This practical uses generated numerical data so it can run without downloading
an external dataset. It trains only on normal observations and tests on normal
and artificially generated anomalous observations.

Code:
See practical_4_autoencoder_anomaly.py.

PRACTICAL 5 – CONTINUOUS BAG OF WORDS (CBOW)
--------------------------------------------
Aim:
Implement the CBOW model to predict a target word from surrounding context.

Theory:
CBOW is a Word2Vec architecture used in Natural Language Processing.
It takes surrounding context words as input and predicts the centre/target
word. The Embedding layer represents words as vectors.

Stages:
1. Prepare a small text corpus.
2. Tokenize words and create a vocabulary.
3. Generate context-target pairs.
4. Train an embedding-based neural network.
5. Predict a target word from context.

Code:
See practical_5_cbow.py.
Note: This is a small teaching example. A tiny corpus gives limited model
quality; the prediction is not expected to be linguistically reliable.

PRACTICAL 6 – TRANSFER LEARNING USING MOBILENETV2
-------------------------------------------------
Aim:
Use a pre-trained CNN and transfer learning for image classification.

Theory:
Transfer Learning reuses features learned by a model trained on a large
dataset. MobileNetV2 is pre-trained on ImageNet. Its convolutional base is
frozen initially, and a new classifier is trained for CIFAR-10.

Stages:
1. Load CIFAR-10 and resize images to 96 x 96.
2. Load ImageNet-pretrained MobileNetV2 without its original classifier.
3. Freeze the pre-trained base.
4. Add GlobalAveragePooling2D and Dense classifier layers.
5. Train and evaluate.
6. Fine-tune selected upper layers if required.

Important distinction:
The supplied code performs image classification, not full object detection.
True object detection also predicts bounding boxes and object locations.

Code:
See practical_6_transfer_learning_mobilenetv2.py.

IMPORTANT VIVA QUESTIONS
------------------------
1. What is Deep Learning?
   A subset of Machine Learning using neural networks with multiple layers.

2. What is an epoch?
   One complete pass through the training dataset.

3. What is batch size?
   Number of samples processed before a model weight update.

4. What is an optimizer?
   An algorithm that updates weights to reduce the loss.

5. What is SGD?
   Stochastic Gradient Descent; an optimizer that updates weights using
   gradients calculated from training samples/batches.

6. What is an activation function?
   A function that adds non-linearity, e.g. ReLU, sigmoid and softmax.

7. What is CNN?
   Convolutional Neural Network, commonly used for image tasks.

8. What is pooling?
   An operation that reduces feature-map spatial dimensions.

9. What is an Autoencoder?
   A neural network trained to reconstruct its input.

10. What is latent space?
    The compressed representation produced by an encoder.

11. What is reconstruction error?
    Difference between the original input and reconstructed output.

12. What is NLP?
    Natural Language Processing; techniques for processing human language.

13. What is CBOW?
    A Word2Vec model that predicts a target word from context words.

14. What is Transfer Learning?
    Reusing knowledge from a pre-trained model for a new task.

15. What does freezing a layer mean?
    Its weights are not updated during training.

16. What is fine-tuning?
    Unfreezing selected pre-trained layers and training them with a small
    learning rate on the new task.

17. What is ImageNet?
    A large image dataset commonly used to pre-train image-recognition models.

18. What is the difference between classification and object detection?
    Classification predicts image categories; object detection also locates
    objects, usually with bounding boxes.

COMMON KERAS FUNCTIONS
----------------------
Sequential()       Create a sequential model
Dense()            Fully connected layer
Conv2D()           Convolution layer
MaxPooling2D()     Pooling layer
Flatten()          Flatten feature maps
compile()          Configure optimizer, loss and metrics
fit()              Train model
evaluate()         Evaluate model
predict()          Generate predictions
