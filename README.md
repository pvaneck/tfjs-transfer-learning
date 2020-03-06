# Transfer Learning in TensorFlow.js

Examples of transfer learning using TensorFlow.js. Here, `tfjs-node` will be used.


## Introduction

Modern, state-of-the-art models typically have millions of parameters and can take inordinate amounts of time
to fully train. **Transfer learning** shortcuts a lot of this training work by taking a model
trained on one task and repurposing it for a second related task. We do this by removing the final layer(s) of the
pre-trained model and then train a new, much smaller model on top of the output of the truncated model. A major
advantage of this technique is that much less training data is needed to train an effective model for new classes.

However, take note that in order for this to be effective, model features learned from the first task should be
general. That is features should be suitable for both the first and second tasks.

To illustrate this in TensorFlow.js, we will take a pre-trained model and build a new classifier on top of it.
