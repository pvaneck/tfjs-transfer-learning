# Transfer Learning in TensorFlow.js

Examples of transfer learning using TensorFlow.js. Here, `tfjs-node` will be used.


## Introduction

Modern, state-of-the-art models typically have millions of parameters and can take inordinate amounts of time
to fully train. **Transfer learning** shortcuts a lot of this training work by taking a model
trained on one task and repurposing it for a second related task. We do this by removing the final layer(s) of the
pre-trained model and then train a new, much smaller model on top of the output of the truncated model. A major
advantage of this technique is that much less training data is needed to train an effective model for new classes.

However, take note that in order for this to be effective, model features learned from the first task should be
general. That is, the features should be suitable for both the first and second tasks.

To illustrate this in TensorFlow.js, we will use the [Fashion-MNIST](https://developer.ibm.com/exchanges/data/all/fashion-mnist/)
dataset.


## Instructions

1. Download and extract the [Fashion-MNIST](https://developer.ibm.com/exchanges/data/all/fashion-mnist/)
   dataset into the root of the project. Your structure should look like this:

    ```shell
    tfjs-transfer-learning
    │
    └───fashion-mnist
        │   fashion-mnist_train.csv
        │   fashion-mnist_test.csv
    ```

1. Train a model from scratch using the first five classes of the dataset. After training, the model will be
   saved in `./fashion-mnist-tfjs`.

    ```shell
    node train-model.js
    ```

1. Train a classifier for the other five classes using transfer learning. However, this time, using
   only a fraction of the training dataset for each class. Training should be much faster now, and test
   accuracy should be similar to the first model.

    ```shell
    node transfer-learn.js
    ```
