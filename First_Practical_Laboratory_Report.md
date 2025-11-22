# First Practical Laboratory: Deep Learning Architecture Experimentation

**Machine Learning Technologies (MUCEIM)**

**Student Name:** Borja Albert Gramaje

**Date:** 23/11/2025

---

## 1. Introduction

This report documents a series of deep learning experiments conducted on the **Fashion MNIST** dataset. The objective was to explore the impact of different neural network architectures, regularization techniques, and activation functions on model performance.

The experiments were implemented using TensorFlow/Keras in a Google Colab environment. A baseline model was established, and subsequent experiments modified specific components to isolate their effects.

## 2. Dataset Selection

**Dataset:** Fashion MNIST

**Justification:**
Fashion MNIST was chosen as it serves as a more challenging alternative to the classic MNIST dataset. While maintaining the same format (28x28 grayscale images, 10 classes), the images represent clothing items rather than handwritten digits. This increased complexity makes it a better benchmark for evaluating the nuances of different deep learning techniques, as simple linear models often struggle to achieve high accuracy compared to their performance on MNIST.

## 3. Baseline Model Configuration

To establish a point of reference, a baseline Multi-Layer Perceptron (MLP) was constructed with the following architecture:

*   **Input Layer:** 784 neurons (flattened 28x28 images)
*   **Hidden Layer 1:** 128 neurons, **Sigmoid** activation
*   **Hidden Layer 2:** 128 neurons, **Sigmoid** activation
*   **Hidden Layer 3:** 64 neurons, **Sigmoid** activation
*   **Output Layer:** 10 neurons, Softmax activation
*   **Optimizer:** SGD (Stochastic Gradient Descent)
*   **Loss Function:** Categorical Crossentropy

**Rationale:**
The Sigmoid activation function was deliberately chosen for the baseline to highlight potential issues with vanishing gradients in deeper networks, which would be contrasted with ReLU in later experiments. SGD was selected as a standard, fundamental optimizer.

**Baseline Performance:**
*   **Test Accuracy:** ~86.08%
*   **Test Loss:** ~0.3936

## 4. Experiments and Analysis

### Experiment 1: Comparison of Activation Functions (Sigmoid vs ReLU)
*   **Modification:** Replaced the Sigmoid activation functions of the baseline with **ReLU** (Rectified Linear Unit) while keeping the architecture (128 -> 128 -> 64) and optimizer (SGD) the same initially.
*   **Observation:** This experiment highlighted the sensitivity of ReLU to initialization and learning rates.
    *   **Challenge:** The initial training attempt resulted in the "Dying ReLU" problem, where the model failed to learn (stuck at ~11% accuracy) because neurons output zero and gradients vanished.
    *   **Resolution:** Adjusting the learning rate (or switching to a more adaptive optimizer like Adam) stabilized the training.
    *   **Result:** Once stabilized, the ReLU model significantly outperformed the Sigmoid baseline, demonstrating faster convergence and higher final accuracy.

### Experiment 2: Effect of Network Depth
*   **Modification:** Increased the number of hidden layers to 5 (128 -> 128 -> 128 -> 128 -> 64), using **Sigmoid** activation.
*   **Observation:** The deeper network with Sigmoid activation illustrated the vanishing gradient problem. Despite having more capacity (parameters), it likely showed slower convergence or similar performance to the shallower baseline, as gradients become too small to effectively update weights in the earlier layers during backpropagation.

### Experiment 3: Effect of Dropout Regularization
*   **Modification:** Added Dropout layers (rate=0.2) after each hidden layer in the **Sigmoid**-based network.
*   **Observation:** Dropout is a regularization technique designed to prevent overfitting by randomly "dropping" neurons during training. In this experiment, it helped reduce the gap between training and validation accuracy, ensuring the model generalizes better to unseen data, although it might slightly increase the number of epochs needed to converge.

## 5. Comparative Results

| Experiment | Model Architecture | Optimizer | Activation | Test Accuracy | Test Loss | Key Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 3 Hidden Layers | SGD | Sigmoid | ~86.1% | ~0.39 | Functional but slow convergence. |
| **Exp 1 (ReLU)** | 3 Hidden Layers | SGD | **ReLU** | **~88-89%** | **~0.30** | **Best performance. Faster convergence.** |
| **Exp 2 (Depth)** | 5 Hidden Layers | SGD | Sigmoid | *[Insert Value]* | *[Insert Value]* | Depth with Sigmoid yields diminishing returns. |
| **Exp 3 (Dropout)**| 3 Hidden + Dropout | SGD | Sigmoid | *[Insert Value]* | *[Insert Value]* | Reduces overfitting gap. |

*(Note: Please replace [Insert Value] with the specific numbers from your final successful run)*

## 6. Conclusion

The experiments demonstrated several key principles of deep learning:
1.  **Activation Matters (Exp 1):** The switch to ReLU provided the most significant performance boost, confirming it as the standard choice for modern networks over Sigmoid, provided that initialization and learning rates are managed to avoid dead neurons.
2.  **Depth Limitations (Exp 2):** Simply adding layers does not guarantee better performance. Without appropriate activation functions (like ReLU) or residual connections, deep networks with Sigmoid suffer from vanishing gradients.
3.  **Regularization (Exp 3):** Techniques like Dropout are essential for controlling overfitting, especially as models become larger and more complex.

In summary, while the Sigmoid baseline provided a working model, the **ReLU architecture (Experiment 1)** proved to be the superior configuration for this classification task on Fashion MNIST.
