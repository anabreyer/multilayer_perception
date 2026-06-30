# Machine Learning Project — Multilayer Perceptron

> This project is an introduction to artificial neural networks, with the implementation of a multilayer perceptron.
> **Version: 5.1**

---

## Table of Contents

- [Chapter I — Introduction](#chapter-i--introduction)
  - [I.1 A bit of history](#i1-a-bit-of-history)
  - [I.2 Multilayer perceptron](#i2-multilayer-perceptron)
  - [I.3 Perceptron](#i3-perceptron)
- [Chapter II — Objectives](#chapter-ii--objectives)
- [Chapter III — General instructions](#chapter-iii--general-instructions)
- [Chapter IV — Mandatory part](#chapter-iv--mandatory-part)
  - [IV.1 Foreword](#iv1-foreword)
  - [IV.2 Dataset](#iv2-dataset)
  - [IV.3 Implementation](#iv3-implementation)
  - [IV.4 Submission](#iv4-submission)
- [Chapter V — Bonus part](#chapter-v--bonus-part)
- [Chapter VI — Submission and Peer-Evaluation](#chapter-vi--submission-and-peer-evaluation)

---

## Chapter I — Introduction

In the language of your choice, you are going to implement a `multilayer perceptron` in order to predict whether a cancer is malignant or benign based on a dataset of breast cancer diagnoses in Wisconsin.

### I.1 A bit of history

Machine learning is a vast field in which artificial neural networks are only a small subset. Nevertheless, we are going to tackle it since it is a really powerful tool that resurfaced a few years ago.

Contrary to what one may think, artificial neural networks have existed for a long time. In his 1948 paper *Intelligent Machinery*, Alan Turing introduced a type of neural network called `B-type unorganised machine`, which he considered the simplest possible model of the nervous system.

The perceptron was invented by Frank Rosenblatt in 1957. It is a single-layer linear classifier and also one of the first neural networks to be implemented. Unfortunately, the results were not as good as expected, and the idea was abandoned. A bit more than 10 years later, the algorithm was improved as the `multilayer perceptron` and was used once again.

### I.2 Multilayer perceptron

The `multilayer perceptron` is a feedforward network (meaning that the data flows from the input layer to the output layer) defined by the presence of one or more hidden layers, as well as an interconnection of all the neurons of one layer to the next.

```
Inputs      Hidden layer   Hidden layer   Output layer
  x0 ──┐
  x1 ──┼── [l0] ──── [l1] ──── [l2] ──── [l3] ── y0
  x2 ──┤                                        ── y1
  x3 ──┘
```

The diagram above represents a network containing 4 dense layers (also called fully connected layers). Its inputs consist of 4 neurons and its output consists of 2 (perfect for binary classification). The weights of one layer to the next are represented by two-dimensional matrices noted **W**_lj lj+1. The matrix **W**_l0l1 is of size (3, 4) for example, as it contains the weights of the connections between layer l0 and layer l1.

The bias is often represented as a special neuron which has no inputs and an output always equal to 1. Like a perceptron, it is connected to all the neurons of the following layer. The bias is generally useful as it allows one to "control the behavior" of a layer.

### I.3 Perceptron

The perceptron is the type of neuron that the `multilayer perceptron` is composed of. It is defined by the presence of one or more input connections, an activation function, and a single output. Each connection contains a weight (also called a parameter) which is learned during the training phase.

Two steps are necessary to get the output of a neuron:

**Step 1 — Weighted sum:**

$$\text{weighted sum} = \sum_{k=0}^{N-1} (x_k \cdot w_k) + \text{bias}$$

**Step 2 — Activation function:**

Apply an activation function on the weighted sum. The output can be understood as the threshold above which the neuron is activated. Some of the most frequently used activation functions: **sigmoid**, **hyperbolic tangent**, and **rectified linear unit (ReLU)**.

---

## Chapter II — Objectives

The goal of this project is to give you a first approach to artificial neural networks, and to have you implement the algorithms at the heart of the training process. At the same time, you are going to have to get reacquainted with the manipulation of derivatives and linear algebra, as they are indispensable mathematical tools for the success of the project.

---

## Chapter III — General instructions

- This project will only be evaluated by humans. You are free to organize and name your files as you desire while respecting the restrictions listed below.
- You are free to use whatever language you want; you have no restrictions on that point.
- **No libraries handling the implementation of artificial neural networks or the underlying algorithms are allowed**; you must code everything from scratch. You can, however, use libraries to handle linear algebra and to display the learning curves.
- In the case of a compiled language, you must submit a Makefile. This Makefile must compile the project, and must contain the usual compilation rules. It should recompile and relink the program only as necessary. The dependencies should also be downloaded/installed with the Makefile as needed.
- The norm is not applied to this project. Nevertheless, you will be asked to be clear and structured in the conception of your source code.

---

## Chapter IV — Mandatory part

### IV.1 Foreword

A non-negligible part of the evaluation will be based on your understanding of the training phase (also called the learning phase) and the underlying algorithms. You will be asked to explain to your corrector the notions of `feedforward`, `backpropagation`, and `gradient descent`. Points will be awarded depending on the clarity of your explanations. These notions are important for the next projects in the branch and will represent a real asset if you wish to continue in this field.

### IV.2 Dataset

The dataset is provided in the resources. It is a csv file with 32 columns, the `diagnosis` column being the label you want to predict, given all the other features of an example. It can be either the value `M` or `B` (for malignant or benign).

The features of the dataset describe the characteristics of a cell nucleus of breast mass extracted with fine-needle aspiration.

As you will see, there is an important data understanding phase before starting to implement the algorithm that will be able to classify it. A good practice would be to begin by exploring the dataset, displaying it with graphs, visualizing, and manipulating its different features.

You have to **separate your dataset into two parts yourself**: one for training and one for validation.

> ⚠️ **The data is raw and should be preprocessed before being used for the training phase.**

### IV.3 Implementation

Your neural network implementation must contain **at least two hidden layers** by default.

The idea is to make you write a program that is a bit more modular (you can use a file or directly provide arguments).

**Example using a config file:**

```
network = model.createNetwork([
    layers.DenseLayer(input_shape, activation='sigmoid'),
    layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
    layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
    layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
    layers.DenseLayer(output_shape, activation='softmax', weights_initializer='heUniform')
])

model.fit(network, data_train, data_valid, loss='categoricalCrossentropy',
          learning_rate=0.0314, batch_size=8, epochs=84)
```

**Example using arguments:**

```bash
python train.py --layer 24 24 24 --epochs 84 --loss categoricalCrossentropy \
  --batch_size 8 --learning_rate 0.0314
```

You must also implement the **softmax** function on the output layer in order to obtain the output as a probabilistic distribution.

In order to evaluate the performance of your model in a robust way during training, you will split your dataset into two parts: one for training and one for validation.

You will also implement **two learning curve graphs** displayed at the end of the training phase (you are free to use any library you want for this purpose): one for **Loss** and one for **Accuracy**.

### IV.4 Submission

You will submit **three programs**:

1. A program to **separate the dataset** into two parts, one for training and the other for validation
2. A **training program**
3. A **prediction program**

*(or you can submit a single program with an option to switch between the three phases)*

To visualize your model's performance during training, you will display the training and validation metrics at each epoch. For example:

```
python mlp.py --dataset data_training.csv
x_train shape : (342, 30)
x_valid shape : (85, 30)
epoch 01/70 - loss: 0.6882 - val_loss: 0.6788
...
epoch 39/70 - loss: 0.0750 - val_loss: 0.0406
epoch 40/70 - loss: 0.0749 - val_loss: 0.0404
epoch 41/70 - loss: 0.0747 - val_loss: 0.0400
...
epoch 70/70 - loss: 0.0640 - val_loss: 0.0474
> saving model './saved_model.npy' to disk...
```

- **Separate program**: You are allowed to use a seed to obtain a repeatable result, because many random factors come into play (such as the weights and bias initialization).
- **Training program**: Will use backpropagation and gradient descent to learn on the training dataset and will save the model (network topology and weights) at the end of its execution.
- **Prediction program**: Will load the weights learned in the previous phase, perform a prediction on a given set, then evaluate it using the **binary cross-entropy error function**:

$$E = -\frac{1}{N} \sum_{n=1}^{N} \left[ y_n \log p_n + (1 - y_n) \log(1 - p_n) \right]$$

---

## Chapter V — Bonus part

The bonus part will be evaluated only if the mandatory part is perfectly done. You are free to implement any functionalities that you think could be interesting. Here is a non-exhaustive list of bonuses:

- A more complex optimization function (e.g. Nesterov momentum, RMSprop, Adam, ...)
- A display of multiple learning curves on the same graph (really useful to compare different models)
- A history of the metrics obtained during training
- The implementation of early stopping
- Evaluate the learning phase with multiple metrics

> 🚨 **The bonus part will only be assessed if the mandatory part is PERFECT.** Perfect means the mandatory part has been completed in its entirety and works without malfunctioning. If you have not passed ALL the mandatory requirements, your bonus part will not be evaluated at all.

---

## Chapter VI — Submission and Peer-Evaluation

Turn in your assignment in your Git repository as usual. Only the work inside your repository will be evaluated during the defense. Don't hesitate to double-check the names of your folders and files to ensure they are correct.
