# Naive Bayes Classifier from Scratch

This repository contains a Python implementation of the Naive Bayes classifier, built from scratch using NumPy. The code demonstrates a fundamental understanding of the Naive Bayes algorithm and its application in machine learning.

## Introduction

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' Theorem. It's widely used for classification tasks, particularly in text classification, spam filtering, and sentiment analysis. Despite its simplifying assumption of feature independence (the "naive" part), Naive Bayes often performs surprisingly well in practice and is computationally efficient.

## Algorithm Overview

The Naive Bayes algorithm works as follows:

1.  **Training Phase (`fit` method):**
    *   Calculate the prior probability of each class based on the training data.
    *   For each class, calculate the mean and variance of each feature.

2.  **Prediction Phase (`predict` method):**
    *   For a given input sample, calculate the posterior probability for each class using Bayes' Theorem. The likelihood is calculated using the probability density function (PDF), assuming a Gaussian (normal) distribution for the features.
    *   The class with the highest posterior probability is assigned as the predicted class.

## Code Structure

The core of the implementation is the `NaiveBayes` class, which contains the following methods:

*   `fit(X, y)`: Trains the classifier on the input data `X` (features) and `y` (labels).
*   `predict(X)`: Predicts the class labels for the input data `X`.
*   `_predict(x)`: A helper function to predict the class label for a single sample `x`.
*   `_pdf(class_idx, x)`: A helper function that calculates the Gaussian probability density function (PDF) for a given sample `x` and class index `class_idx`.

## Dependencies

*   NumPy: For numerical operations and array handling.
*   Scikit-learn: Used for generating a sample dataset and splitting it into training and testing sets (only for demonstration purposes in the `__main__` block).

## Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/muhammadhazrat/ML_Naive_Bayes/tree/main/Naive_Bayes_From_Scratch
    cd Naive_Bayes_From_Scratch
    ```

2.  **Run the script:**

    ```bash
    python naive_bayes.py
    ```
    The output accuracy depends on the random state and dataset that is used. A sample output is shown below.

    ```bash
    Naive Bayes Accuracy: 0.965
    ```

    This will execute the example code in the `__main__` block, which demonstrates how to use the `NaiveBayes` class:
    *   Generates a synthetic classification dataset using `sklearn.datasets.make_classification`.
    *   Splits the data into training and testing sets using `sklearn.model_selection.train_test_split`.
    *   Creates an instance of the `NaiveBayes` class.
    *   Trains the classifier using the `fit` method.
    *   Makes predictions on the test set using the `predict` method.
    *   Calculates and prints the accuracy of the classifier.

## Key Features

*   **Clear and concise implementation:** The code is well-structured and easy to understand, making it a great resource for learning about the Naive Bayes algorithm.
*   **From-scratch implementation:**  The code avoids using high-level machine learning libraries for the core algorithm, providing a deeper understanding of the underlying principles.
*   **NumPy-based:** Leverages NumPy for efficient numerical computations.

## Limitations

*   **Naive Assumption:** Assumes that features are independent, which may not always hold true in real-world datasets.
*   **Gaussian Assumption:** Assumes that features follow a Gaussian distribution. This might not be suitable for all types of data.

## Further Improvements

*   **Handling other distributions:** Extend the code to handle other probability distributions (e.g., Bernoulli, Multinomial) for different types of features.
*   **Add smoothing techniques:** Implement Laplace smoothing or other smoothing techniques to handle zero probabilities, especially when dealing with limited training data.
*   **Cross-validation:** Incorporate cross-validation to get a more robust estimate of the model's performance.

## Conclusion

This project provides a solid foundation for understanding and implementing the Naive Bayes algorithm. It's a valuable resource for anyone interested in learning about probabilistic machine learning and building classifiers from scratch. Feel free to fork this repository, experiment with the code, and contribute to its improvement!
