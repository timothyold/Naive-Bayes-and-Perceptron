# Naive Bayes and Perceptron Algorithms

This repository contains two machine learning models implemented in Python each with their own dataset.

## Datasets

### 1. `lab02_dataset_1.csv`
This is a CSV file with three columns: `X`, `Y`, and `Z`.
### 2. `lab02_dataset_2.xlsx`
This is an Excel file with various columns representing different attributes, including `ID`, `KIDSDRIV`, `AGE`, `HOMEKIDS`, `YOJ`, `INCOME`, `HOME_VAL`, `TRAVTIME`, `BLUEBOOK`, `TIF`, `MVR_PTS`, `CAR_AGE`, `CLM_AMT`, `CLM_COUNT`, and `EXPOSURE`.

### Machine Learning Models

## Naive Bayes Algorithm Detailed Description

The `NaiveBayes.py` file contains a function `naive_bayes_vehicle_usage` that implements a Naive Bayes classifier using the `CategoricalNB` class from `sklearn`. Here is how the algorithm works step by step:

1. **Data Loading**: The function starts by loading a dataset from the provided `datasetpath`, which is expected to be an Excel file.

2. **Data Preprocessing**: 
    - The dataset is expected to have columns like `CAR_USE`, `CAR_TYPE`, `OCCUPATION`, and `EDUCATION`. These columns are extracted, and any rows with missing values are dropped.
    - The `CAR_USE` column is encoded into numerical labels using `LabelEncoder`. This column will be used as the target variable (`y`) for the model.
    - The other columns (`CAR_TYPE`, `OCCUPATION`, `EDUCATION`) are also encoded numerically, converting categorical data into a format suitable for the Naive Bayes model.

3. **Training and Test Split**: The preprocessed data is then split into training and test sets, with 80% of the data used for training and the remaining 20% for testing.

4. **Model Training**:
    - A `CategoricalNB` model is instantiated with `alpha=0.01`, which is a smoothing parameter.
    - The model is trained (`fit`) using the training data.

5. **Prediction and Evaluation**:
    - The trained model is used to predict the `CAR_USE` category on the test set.
    - The accuracy of the model is calculated by comparing the predicted and actual values of `CAR_USE` in the test set.
    - A misclassification rate is also calculated as `1 - accuracy`.

6. **Visualization**:
    - A histogram is generated showing the distribution of predicted probabilities for the `CAR_USE` category being 'Private'.

7. **Evaluation on Specific Profiles**:
    - The model is also used to predict the probability of specific fictitious profiles being for 'Private' or 'Commercial' use.
    - These profiles are defined within the code with specific attributes for `CAR_TYPE`, `OCCUPATION`, and `EDUCATION`.
    - The probabilities for these profiles are printed out.

The function returns the accuracy and misclassification rate of the model, and prints the probability distributions for the specified profiles.


## Perceptron Algorithm Detailed Description

The `Perceptron.py` file contains an implementation of the Perceptron learning algorithm, which is a type of linear classifier. The process is as follows:

1. **Data Preparation**:
    - The dataset is loaded from `lab02_dataset_1.csv`, which is expected to have at least two columns for features and one column for labels.
    - Features (`X`) are extracted from all but the last column, and labels (`y`) are taken from the last column, where labels are mapped to `1` for 'Positive' and `-1` for 'Negative'.

2. **Bias Term Addition**:
    - A column of ones is added to the feature matrix `X` to incorporate the bias term in the weight vector, allowing the Perceptron to account for the offset from the origin in its decision boundary.

3. **Perceptron Training Function (`my_perceptron`)**:
    - Initializes a weight vector with zeros.
    - Iteratively updates the weight vector based on the training data:
        - For each epoch (pass through the entire dataset), the algorithm iterates over all instances (`xi`, `target`) in the dataset.
        - It calculates the update for the weights based on the difference between the target label and the prediction made by the current weight vector.
        - The learning rate (`learning_rate`) influences the size of the weight update.
        - If the predicted label does not match the actual label, the weights are updated accordingly.
        - The process continues for a specified number of epochs or until the misclassification rate falls below a certain threshold (1% in this case).

4. **Model Training**:
    - The `my_perceptron` function is called with the dataset and parameters to compute the final weight vector that defines the decision boundary of the model.

5. **Visualization**:
    - The algorithm visualizes the dataset and the decision boundary in a 3D plot.
    - Points are colored based on their actual labels, and the decision surface is shown as a plane in the 3D space.

This implementation of the Perceptron algorithm demonstrates a fundamental approach to binary classification in machine learning, where the decision boundary linearly separates the data points into two categories based on the learned weights.


## Usage

To use these models, you need a Python environment with necessary libraries installed (e.g., `numpy`, `pandas`). Load the datasets using `pandas` and apply the models to perform classification tasks.

