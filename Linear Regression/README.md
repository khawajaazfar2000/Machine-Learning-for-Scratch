# Linear Regression from Scratch

This repository contains a basic implementation of the Linear Regression algorithm from scratch in Python, along with an example demonstrating its usage and performance on a synthetic dataset.

## Table of Contents

* [Introduction](#introduction)
* [Files in this Repository](#files-in-this-repository)
* [Algorithm Overview](#algorithm-overview)
* [How to Run](#how-to-run)
* [Dependencies](#dependencies)


## Introduction

Linear Regression is a fundamental supervised learning algorithm used for modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data. This project provides a clear and concise implementation of Linear Regression using gradient descent, built entirely from basic NumPy operations, offering insight into the algorithm's inner workings.

## Files in this Repository

* `LinearRegression.py`: This file contains the core implementation of the Linear Regression model.
    * `__init__(self, lr=0.001, n_inters=1000)`: Initializes the model with a learning rate (`lr`) and number of iterations (`n_inters`) for gradient descent.
    * `fit(self, X, y)`: Trains the model using the provided training data `X` and target `y`. It iteratively updates the `weights` and `bias` using the gradient descent optimization algorithm.
    * `predict(self, X)`: Makes predictions on new data `X` using the learned `weights` and `bias`.
* `train.py`: This file demonstrates how to use the `LinearRegression` class.
    * Generates a synthetic dataset for regression using `sklearn.datasets.make_regression`.
    * Splits the data into training and testing sets.
    * Initializes and trains the `LinearRegression` model.
    * Calculates and prints the Mean Squared Error (MSE) of the predictions on the test set.
    * Visualizes the original data points, training points, test points, and the fitted regression line using `matplotlib`.

## Algorithm Overview

The `LinearRegression.py` implements the following steps:

1.  **Initialization**: Weights are initialized to zeros, and the bias is initialized to zero.
2.  **Prediction**: For each iteration, the model calculates predictions ($$\hat{y}$$) using the current weights ($$w$$) and bias ($$b$$):
    $$ \hat{y} = X \cdot w + b $$
3.  **Cost Function**: The Mean Squared Error (MSE) is implicitly minimized.
4.  **Gradient Calculation**: The gradients of the cost function with respect to weights ($$dw$$) and bias ($$db$$) are calculated:
    $$ dw = \frac{1}{N} \cdot X^T \cdot (\hat{y} - y) $$
    $$ db = \frac{1}{N} \cdot \sum (\hat{y} - y) $$
    where $$N$$ is the number of samples.
5.  **Parameter Update**: Weights and bias are updated using the learning rate ($$lr$$) and the calculated gradients:
    $$ w = w - lr \cdot dw $$
    $$ b = b - lr \cdot db $$
6.  **Iteration**: Steps 2-5 are repeated for a specified number of iterations (`n_inters`).

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/khawajaazfar2000/Machine-Learning-for-Scratch/tree/e93cc13f44df51e14a9b6d21802a6dded72b1149/Linear%20Regression
    cd Linear Regression
    ```
2.  **Run the training script:**
    ```bash
    python train.py
    ```
    This will execute the `train.py` script, which will train the Linear Regression model, make predictions, calculate MSE, and display a scatter plot with the fitted regression line.

## Dependencies

You'll need the following Python libraries installed:

* `numpy`
* `scikit-learn`
* `matplotlib`

You can install them using pip:

```bash
pip install numpy scikit-learn matplotlib
