# ALS_training.py:
##### For this code we need 25Millions Movielens dataset which can be find at https://grouplens.org/datasets/movielens/25m/

Recommendation System Training Using Matrix Factorization

This Python script is designed to train a recommendation system model using matrix factorization. The model includes user and item biases as well as latent factors for improved accuracy.
Prerequisites

Before using this script, make sure you have the following:

    Python installed on your system.
    Required Python packages (NumPy, Pandas, Matplotlib, tqdm).
    A dataset in CSV format (provided as 'ratings.csv' in this example).

Usage

    Make sure you have the necessary dataset in CSV format, and update the 'ratings.csv' file path as needed.

    Run the script to perform the following steps:
        Data Preprocessing: Load the data, split it into training and testing sets, and create sparse user-movie matrices.
        User and Item Biases Training: Train the model with user and item biases using stochastic gradient descent.
        Bias and Latent Vectors Training: Train the model with user and item biases and latent factors using stochastic gradient descent.
        Evaluation: Calculate and visualize the training and testing RMSE (Root Mean Square Error).

    The script will save the trained model and evaluation results as 'saved_data.pkl' and 'bias-only_data.pkl' for later use.

Parameters

You can adjust various parameters in the script, including the number of latent factors, regularization parameters, and the number of training iterations. These parameters can impact the model's performance, so you may want to experiment with different values.
Data Saved

The script saves the following data:

    User and item latent factors.
    User and item biases.
    Iteration numbers.
    Training and testing loss values.
    Training and testing RMSE values.

data_begin.pkl: file containing the sparse matrices
bias-only_data.pkl: for the bias only training, it contains:
    User and item latent factors.
    User and item biases.
    Iteration numbers.
    Training and testing loss values.
    Training and testing RMSE values.
saved_data_16.pkl: for the bias, u, v training and 16 latent dimension:
    User and item latent factors.
    User and item biases.
    Iteration numbers.
    Training and testing loss values.
    Training and testing RMSE values.

You can use this saved data to analyze the model's performance or use it to make recommendations.
Plotting and Visualization

The script provides options to visualize the training process and the RMSE values over iterations. The generated plots are saved as image files for reference.

# recommender.py
Python script that loads data, performs movie recommendations using a collaborative filtering approach, and plots a 2D embedding of movie items.
