Fashion-MNIST Classification Using TensorFlow & PyTorch
This repository contains implementations of neural network models trained on the Fashion-MNIST dataset using TensorFlow and PyTorch. The goal is to showcase and compare how the same task can be solved in both frameworks, including best practices such as normalization, dropout, and efficient training loops.

Project Overview
Fashion-MNIST is a dataset of 28x28 grayscale images of 10 different clothing categories (e.g., T-shirts, trousers, bags, shoes). It is a commonly used benchmark for image classification.

In this project, two models are built and trained:

TensorFlow implementation: Uses tf.keras.Sequential API with dense layers, batch normalization, dropout, and ReLU activation.

PyTorch implementation: Uses a custom nn.Module class with similar architecture and a manual training loop utilizing accelerate and tqdm libraries for performance and progress tracking.

Dataset
Source: Fashion-MNIST

Data splits: Training set (60,000 images), Test set (10,000 images)

Preprocessing: Pixel values normalized from [0,255] to [0,1] using transforms.ToTensor() (PyTorch) or Rescaling layer (TensorFlow).

Model Architecture
Both implementations share the same general structure:

Input layer for 28x28 grayscale images

Flatten layer to convert images to 1D vectors (28*28 = 784 features)

Dense layer with 300 units + Batch Normalization + ReLU

Dense layer with 100 units + Batch Normalization + ReLU + Dropout(0.2)

Dense layer with 50 units + Batch Normalization + ReLU + Dropout(0.2)

Output layer with 10 units (one per class)

Loss function: Sparse categorical cross-entropy (TensorFlow) / CrossEntropyLoss (PyTorch), which internally applies Softmax

Training Details
Optimizer: Stochastic Gradient Descent (SGD) with learning rate 0.01, momentum 0.9, and Nesterov momentum enabled

Batch size: 32

Epochs: 5

PyTorch enhancements: Uses the accelerate library to optimize device placement and multi-GPU compatibility, and tqdm for progress bars

Validation: Accuracy is computed on a validation split during training to monitor performance

Results
Both frameworks achieve comparable accuracy on validation and test sets (usually around ~85-90% accuracy)

The PyTorch model uses a custom training loop, giving more control and insight into the training process

The TensorFlow model uses the model.fit() API for simplicity and faster prototyping



Fashion-MNIST_PyTorch.ipynb

Future Work
Implement Convolutional Neural Networks (CNNs) for improved accuracy

Add learning rate schedulers and advanced optimizers

Explore data augmentation techniques

Compare training speed and memory consumption between frameworks

Contact
If you have questions or feedback, feel free to contact me at leman.qasimli13@gmail.com or open an issue on GitHub.

This project is a learning resource to understand deep learning workflows in TensorFlow and PyTorch.
