# Handwritten Digit Recognition using Kernel Perceptron

## Overview

In this project, we implement the Kernel Perceptron method from scratch to perform multiclass classification on the MNIST dataset using the One-vs-All approach. The prediction is carried out using a varying number of epochs and degrees of polynomial. Two distinct predictors are employed for the predictions:

1. The average of all the predictors.
2. The predictor with the smallest training error.

Both predictors demonstrate similar performance across different epochs and polynomial degrees. The optimal results are obtained with a polynomial degree of 2 and epochs set to 6 or 8. Conversely, the worst performance is observed at a polynomial degree of 3 and epoch 1.

## Introduction

Linear predictors are commonly used in classification problems due to their practicality. However, in real-world scenarios, the relationship between the features and the target variable is rarely linear, leading to high bias. To address this, techniques such as feature expansion are employed, where higher-level features are derived from existing features and incorporated into the feature vector.

The main objective is to identify a hyperplane that separates different classes in the expanded dimensional space, as illustrated in Figure 1. This approach allows for the learning of more complex predictors, such as circles and parabolas, in the original feature space. However, the risk of overfitting increases with the dimensionality of the feature space.

 ![](https://github.com/RishavMondal/Multiclass_Classification_Kernel_perceptron/blob/main/Screenshot%202024-08-09%20123421.png)

One of the significant drawbacks of this method is the increased computational cost associated with higher dimensionality, as calculating the coordinates of each point in the augmented space becomes more complex. However, by utilizing kernels, this problem can be mitigated, reducing complexity while achieving similar results.

---

