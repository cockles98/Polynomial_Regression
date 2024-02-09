# Polynomial Regression Analysis

This repository contains Python code to perform polynomial regression analysis using scikit-learn. The analysis includes generating noisy data, fitting polynomials of varying degrees, evaluating model performance, and analyzing the effect of regularization.

## Installation

To run the code in this repository, ensure you have Python installed, along with the following dependencies:

- numpy
- matplotlib
- scikit-learn

You can install the dependencies using pip:

```python
pip install numpy matplotlib scikit-learn
```

## Description
1. Data Generation:
The code generates noisy data based on a given analytical function. The function generate_noisy_data adds Gaussian noise to the analytical function.

3. Polynomial Regression:
The script performs polynomial regression using both vanilla polynomial regression and Ridge regression. The polynomial_regression function fits a polynomial of a specified degree to the data.

4. Model Evaluation:
Model performance is evaluated using Mean Squared Error (MSE). The evaluate_model function splits the data into training and test sets, fits the model, and computes the MSE for both sets.

5. MSE Analysis:
The script analyzes the behavior of MSE for different polynomial degrees and regularization parameters. It provides insights into model complexity and the effect of regularization.

6. Plotting:
The script provides plotting functionality to visualize the generated data, fitted polynomials, and MSE values.

## Results
After running the script, you'll get insights into the best polynomial degree and the effect of regularization on model performance. Plots are provided to visualize the results.

## Usage
The main functionality of the code is encapsulated within the `main()` function in the provided script. By running the script, you'll perform a series of analyses on polynomial regression. You can change the `main` function to set a customize situation for analysis.
