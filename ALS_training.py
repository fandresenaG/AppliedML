# -*- coding: utf-8 -*-
"""
RAMONJISON Tiana Fandresena Gerald
This .py code is for the training for ALS-based modeling
"""

import numpy as np
import pandas as pd
import pickle
import csv
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.linalg import cholesky, inv
from tqdm import tqdm
import tabulate

#Import data 
data = np.loadtxt('ratings.csv',delimiter =',',skiprows = 1)[:,:3].astype('str')

ratings_float = np.array([float(data[i][2]) for i in range(len(data))])
plt.hist(ratings_float)
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.grid()
plt.savefig('hist.jpg')
plt.show()

def sparse_matrix(data, split = 0.9):
    '''Create sparse user-movie matrices and split the data into training and testing sets.

    Args:
        data (list of tuples): Raw user-movie-rating data.
        split (float): Proportion of data to use for training (default is 0.9).

    Returns:
        map_user_index (dict): Maps userId to user index.
        map_index_user (list): Maps user index to userId.
        map_movie_index (dict): Maps movieId to movie index.
        map_index_movie (list): Maps movie index to movieId.
        sparse_user (list of lists): Sparse user-movie matrix.
        sparse_movie (list of lists): Sparse movie-user matrix.
        user_train (list of lists): Training data for users.
        movie_train (list of lists): Training data for movies.
        user_test (list of lists): Testing data for users.
        movie_test (list of lists): Testing data for movies.
    '''

    np.random.shuffle(data)

    map_user_index = {}
    map_index_user = []

    map_movie_index = {}
    map_index_movie = []

    sparse_user = []
    sparse_movie = []

    for i in range(len(data)):
      # Test if the user data[i][0] is already in map_user_index or not
      if data[i][0] not in map_user_index.keys():
        index_user = len(map_index_user)

        map_index_user.append(data[i][0])

        sparse_user.append([])

        map_user_index[data[i][0]] = index_user

      # In case the user in data[i][0] is already in the dict map_user_index
      else:
        # take the index of the user
        index_user = map_user_index[data[i][0]]

      # Test if the movie in data[i][1] is already in map_user_index or not
      if data[i][1] not in map_movie_index.keys():
        index_movie = len(map_index_movie)

        map_index_movie.append(data[i][1])

        sparse_movie.append([])

        map_movie_index[data[i][1]] = index_movie

      # In case the movie in data[i][1] is already in the dict map_movie_index
      else:
        index_movie = map_movie_index[data[i][1]]

      # Add a tuple with the index of the movie and the rating of this movie for the current user
      sparse_user[index_user].append((index_movie, float(data[i][2])))

      # Add a tuple with the index of the user and the rating of the movie that the user put
      sparse_movie[index_movie].append((index_user, float(data[i][2])))

    split_point = split * len(data)

    #Initialization
    user_train = [[] for i in range(len(map_index_user))]
    movie_train = [[] for i in range(len(map_index_movie))]

    user_test = [[] for i in range(len(map_index_user))]
    movie_test = [[] for i in range(len(map_index_movie))]

    #Splitting data into train/test
    for i in range(len(data)):

      user = data[i][0]
      movie = data[i][1]
      rating = float(data[i][2])

      current_user = map_user_index[user]
      current_movie = map_movie_index[movie]

      if i < split_point:
        user_train[current_user].append((current_movie,rating))
        movie_train[current_movie].append((current_user,rating))

      else:
        user_test[current_user].append((current_movie,rating))
        movie_test[current_movie].append((current_user,rating))


    return map_user_index, map_index_user, map_index_movie, map_movie_index, sparse_user, sparse_movie, user_train, movie_train, user_test, movie_test

map_user_index, map_index_user, map_index_movie, map_movie_index, sparse_user, sparse_movie, user_train, movie_train, user_test, movie_test = sparse_matrix(data, split = 0.9)

# This function return two array, the first containing the number of movie for each user
# And the second the number of user who rated each movie
def count_user_movie(sparse_user,sparse_movie):
    """ This function takes two arguments:
  sparse_user: contains all the users and the movies that each user have rated
  sparse_movie: contains all the movies and the users who rated it"""

    # Initializing the 02 lists
    movies_per_user = [] # list of the number of movies each user rated

    users_per_movie = [] # list of number of user who rated each movie

    # Count the movie that each user rated
    for user in sparse_user:
        movies_per_user.append(len(user))

    # Count the user who rated each movie
    for movie in sparse_movie:
        users_per_movie.append(len(movie))

    return movies_per_user, users_per_movie

movies_per_user, users_per_movie = count_user_movie(sparse_user,sparse_movie)

def ploting(movies_per_user, users_per_movie):
    """ This function is for the power law plot
    It takes two arguments:
    movies_per_user: list of the number of movies each user rated
    users_per_movie: list of number of user who rated each movie"""

    # Number of users who have rated the same number of movies
    y_user = []

    # Number of movies who have been rated by same number of users
    y_movie = []

    # Add the number of users who have rated the same number of movie as "user"
    for user in movies_per_user:
        y_user.append(movies_per_user.count(user))

    # Add the number of movies who have been rated by the same number of user as "movie"
    for movie in users_per_movie:
        y_movie.append(users_per_movie.count(movie))

    # Ploting

    plt.scatter(users_per_movie, y_movie, label="Movie")

    plt.scatter(movies_per_user, y_user, label="User")

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.grid()
    plt.legend()

    plt.savefig('power.pdf')
    plt.show()

ploting(movies_per_user, users_per_movie)

# Define a dictionary to store the data you want to save
saved_d = {
    'map_user_index': map_user_index,
    'map_index_user': map_index_user,
    'map_movie_index': map_movie_index,
    'map_index_movie': map_index_movie,
    'sparse_user': sparse_user,
    'sparse_movie': sparse_movie,
    'user_train': user_train,
    'movie_train': movie_train,
    'user_test': user_test,
    'movie_test': movie_test
}

# Specify the file path where you want to save the data
save_path = 'data_begin.pkl'

# Use Pickle to serialize and save the data
with open(save_path, 'wb') as file:
    pickle.dump(saved_d, file)

print("Data saved successfully.")

"""### This part is for the bias only"""

# Initialize user and item biases with zeros
num_users = len(map_user_index)
num_movies = len(map_movie_index)

bias_user = np.zeros(num_users)
bias_item = np.zeros(num_movies)

def update_user_bias(bias_user, bias_item, sparse_user, lambda_g, gamma):
    '''
    Update user biases using the provided sparse user-movie matrix and regularization parameters.

    Args:
        bias_user (numpy.ndarray): Array of user biases to be updated.
        bias_item (numpy.ndarray): Array of item biases.
        sparse_user (list of lists): Sparse user-movie matrix.
        lambda_g (float): Regularization parameter for user biases.
        gamma (float): Regularization parameter for biases.

    Returns:
        bias_user (numpy.ndarray): Updated user biases.
    '''
    num_users = len(bias_user)

    for m in range(num_users):
        bias = 0
        item_counter = 0

        for l, r in sparse_user[m]:
            bias += lambda_g * (r - bias_item[l])
            item_counter += 1

        bias = bias / (lambda_g * item_counter + gamma)
        bias_user[m] = bias

    return bias_user

def update_item_bias(bias_item, bias_user, sparse_movie, lambda_g, gamma):
    '''
    Update item biases using the provided sparse movie-user matrix and regularization parameters.

    Args:
        bias_item (numpy.ndarray): Array of item biases to be updated.
        bias_user (numpy.ndarray): Array of user biases.
        sparse_movie (list of lists): Sparse movie-user matrix.
        lambda_g (float): Regularization parameter for item biases.
        gamma (float): Regularization parameter for biases.

    Returns:
        bias_item (numpy.ndarray): Updated item biases.
    '''
    num_items = len(bias_item)

    for n in range(num_items):
        bias = 0
        item_counter = 0

        for k, r in sparse_movie[n]:
            bias += lambda_g * (r - bias_user[k])
            item_counter += 1

        bias = bias / (lambda_g * item_counter + gamma)
        bias_item[n] = bias

    return bias_item

def calculate_loss(sparse_user, bias_user, bias_item, lambda_g, gamma):
    '''
    Calculate the loss of a recommendation system with user and item biases.

    Args:
        sparse_user (list of lists): Sparse user-movie matrix.
        bias_user (numpy.ndarray): Array of user biases.
        bias_item (numpy.ndarray): Array of item biases.
        lambda_g (float): Regularization parameter for bias terms.
        gamma (float): Regularization parameter for bias terms.

    Returns:
        loss (float): The calculated loss.
    '''
    error_sum = 0

    for m, user_ratings in enumerate(sparse_user):
        for n, rating in user_ratings:
            error_sum += (rating - bias_user[m] - bias_item[n]) ** 2

    loss = -lambda_g * 0.5 * error_sum - gamma * 0.5 * np.dot(bias_user, bias_user) - gamma * 0.5 * np.dot(bias_item, bias_item)

    return loss

def calculate_rmse(sparse_user, bias_user, bias_item):
    '''
    Calculate the Root Mean Square Error (RMSE) of a recommendation system with user and item biases.

    Args:
        sparse_user (list of lists): Sparse user-movie matrix.
        bias_user (numpy.ndarray): Array of user biases.
        bias_item (numpy.ndarray): Array of item biases.

    Returns:
        rmse (float): The calculated RMSE.
    '''
    error_sum = 0
    num_ratings = 0

    for m, user_ratings in enumerate(sparse_user):
        for n, rating in user_ratings:
            error_sum += (rating - bias_user[m] - bias_item[n]) ** 2
            num_ratings += 1

    rmse = np.sqrt(error_sum / num_ratings)

    return rmse

def train_bias_model(user_train, movie_train, user_test, movie_test, map_user_index, map_movie_index,
                     num_latent_factors=5, lambda_g=0.01, gamma=0.1, num_iterations=30):
    """
    Train a matrix factorization model with user and item biases using stochastic gradient descent.

    Args:
        user_train (list of lists): Training data for users (sparse matrix of ratings).
        movie_train (list of lists): Training data for movies (sparse matrix of ratings).
        user_test (list of lists): Testing data for users (sparse matrix of ratings).
        movie_test (list of lists): Testing data for movies (sparse matrix of ratings).
        map_user_index (dict): Maps userId to user index.
        map_movie_index (dict): Maps movieId to movie index.
        num_latent_factors (int): Number of latent factors (default is 5).
        lambda_g (float): Regularization parameter for model parameters (default is 0.01).
        gamma (float): Regularization parameter for bias terms (default is 0.1).
        num_iterations (int): Number of iterations for training (default is 30).

    Returns:
        bias_user (numpy.ndarray): Learned user bias vector.
        bias_item (numpy.ndarray): Learned item bias vector.
        loss_train_history (list): List of training loss values at each iteration.
        rmse_train_history (list): List of RMSE values on the training set at each iteration.
        loss_test_history (list): List of testing loss values at each iteration.
        rmse_test_history (list): List of RMSE values on the testing set at each iteration.
        iteration_numbers (list): List of iteration numbers.
    """
    num_users = len(map_user_index)
    num_items = len(map_movie_index)

    bias_user = np.zeros(num_users)
    bias_item = np.zeros(num_items)

    loss_train_history = []
    loss_test_history = []
    rmse_train_history = []
    rmse_test_history = []
    iteration_numbers = []

    for iteration in tqdm(range(num_iterations)):
        # Update user biases and item biases
        bias_user = update_user_bias(bias_user, bias_item, user_train, lambda_g, gamma)
        bias_item = update_item_bias(bias_item, bias_user, movie_train, lambda_g, gamma)

        # Calculate loss and RMSE for the training set
        loss_train = - calculate_loss(user_train, bias_user, bias_item, lambda_g, gamma)
        loss_train_history.append(loss_train)
        rmse_train = calculate_rmse(user_train, bias_user, bias_item)
        rmse_train_history.append(rmse_train)

        # Calculate loss and RMSE for the testing set
        loss_test = - calculate_loss(user_test, bias_user, bias_item, lambda_g, gamma)
        loss_test_history.append(loss_test)
        rmse_test = calculate_rmse(user_test, bias_user, bias_item)
        rmse_test_history.append(rmse_test)

        iteration_numbers.append(iteration)

    return bias_user, bias_item, loss_train_history, rmse_train_history, loss_test_history, rmse_test_history, iteration_numbers

# Usage example
bias_user, bias_item, loss_train_history, rmse_train_history, loss_test_history, rmse_test_history, iteration_numbers = train_bias_model(user_train, movie_train, user_test, movie_test, map_user_index, map_movie_index,
                     num_latent_factors=10, lambda_g=0.01, gamma=0.1, num_iterations=30)

iteration_numbers = np.arange(1,31)

# Add the first subplot
plt.plot(iteration_numbers, loss_train_history, marker='o')
plt.plot(iteration_numbers, loss_test_history, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Loss (Negative Log Likelihood)')
plt.grid(True)
plt.show()

# Add the second subplot
plt.subplot(1, 2, 2)
plt.plot(iteration_numbers, rmse_train_history, marker='o', label='RMSE train')
plt.plot(iteration_numbers, rmse_test_history, marker='o', label='RMSE test')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.grid(True)

# Save the figure with both subplots to a PDF file
plt.savefig('bias_only.pdf')

# Display the combined figure
plt.show()

# Define a dictionary to store the data you want to save
saved_data_bias = {
    'iteration': iteration_numbers,
    'loss_train': loss_train_history,
    'rmse_train': rmse_train_history,
    'loss_test': loss_test_history,
    'rmse_test': rmse_test_history,
}

# Specify the file path where you want to save the data
save_path = 'bias-only_data.pkl'

# Use Pickle to serialize and save the data
with open(save_path, 'wb') as file:
    pickle.dump(saved_data_bias, file)

print("Data saved successfully.")



"""### This code is for the bias and latent vectors"""

# Parameters
latent_dimension = 16
regularization_lambda = 0.001
regularization_tau = 0.01
regularization_gamma = 0.5
num_iterations = 10

# Initialisation of matrices and vectors
num_users = len(sparse_user)
num_movies = len(sparse_movie)

user_biases = np.zeros(num_users)
movie_biases = np.zeros(num_movies)

user_latent_factors = normal(0, np.sqrt(1/np.sqrt(latent_dimension)), size=[num_users, latent_dimension])
movie_latent_factors = normal(0, np.sqrt(1/np.sqrt(latent_dimension)), size=[num_movies, latent_dimension])

def update_user_latent_factors(user_id, sparse_user, user_biases, movie_biases, movie_latent_factors, regularization_lambda, regularization_tau, latent_dimension):
    """
    Update user latent factors using stochastic gradient descent for matrix factorization.

    Args:
        user_id (int): User index for whom latent factors are updated.
        sparse_user (list of lists): Sparse user-movie matrix.
        user_biases (numpy.ndarray): User bias vector.
        movie_biases (numpy.ndarray): Movie bias vector.
        movie_latent_factors (numpy.ndarray): Movie latent factors.
        regularization_lambda (float): Regularization parameter for model parameters.
        regularization_tau (float): Regularization parameter for latent factors.
        latent_dimension (int): Number of latent factors.

    Returns:
        numpy.ndarray: Updated user latent factors.
    """
    matrix_left = np.zeros((latent_dimension, latent_dimension))
    matrix_right = np.zeros(latent_dimension)

    for movie_id, rating in sparse_user[user_id]:
        movie_vec = movie_latent_factors[movie_id]
        matrix_left += np.outer(movie_vec, movie_vec)
        matrix_right += movie_vec * (rating - user_biases[user_id] - movie_biases[movie_id])

    regularization_term = regularization_tau * np.identity(latent_dimension)
    matrix_left = regularization_lambda * matrix_left + regularization_term

    return np.linalg.solve(matrix_left, regularization_lambda * matrix_right)

def update_movie_latent_factors(movie_id, sparse_movie, user_biases, movie_biases, user_latent_factors, regularization_lambda, regularization_tau, latent_dimension):
    """
    Update movie latent factors using stochastic gradient descent for matrix factorization.

    Args:
        movie_id (int): Movie index for which latent factors are updated.
        sparse_movie (list of lists): Sparse movie-user matrix.
        user_biases (numpy.ndarray): User bias vector.
        movie_biases (numpy.ndarray): Movie bias vector.
        user_latent_factors (numpy.ndarray): User latent factors.
        regularization_lambda (float): Regularization parameter for model parameters.
        regularization_tau (float): Regularization parameter for latent factors.
        latent_dimension (int): Number of latent factors.

    Returns:
        numpy.ndarray: Updated movie latent factors.
    """
    matrix_left = np.zeros((latent_dimension, latent_dimension))
    matrix_right = np.zeros(latent_dimension)

    for user_id, rating in sparse_movie[movie_id]:
        user_vec = user_latent_factors[user_id]
        matrix_left += np.outer(user_vec, user_vec)
        matrix_right += user_vec * (rating - movie_biases[movie_id] - user_biases[user_id])

    regularization_term = regularization_tau * np.identity(latent_dimension)
    matrix_left = regularization_lambda * matrix_left + regularization_term

    return np.linalg.solve(matrix_left, regularization_lambda * matrix_right)

def update_user_biases(user_id, sparse_user, user_latent_factors, movie_latent_factors, movie_biases, regularization_lambda, regularization_gamma):
    """
    Update user biases using stochastic gradient descent for matrix factorization.

    Args:
        user_id (int): User index for whom biases are updated.
        sparse_user (list of lists): Sparse user-movie matrix.
        user_latent_factors (numpy.ndarray): User latent factors.
        movie_latent_factors (numpy.ndarray): Movie latent factors.
        movie_biases (numpy.ndarray): Movie bias vector.
        regularization_lambda (float): Regularization parameter for model parameters.
        regularization_gamma (float): Regularization parameter for biases.

    Returns:
        float: Updated user bias.
    """
    bias = 0

    for movie_id, rating in sparse_user[user_id]:
        bias += rating - (np.dot(user_latent_factors[user_id], movie_latent_factors[movie_id]) + movie_biases[movie_id])

    #user_biases[user_id] = (regularization_lambda * bias) / (regularization_lambda * len(sparse_user[user_id]) + regularization_gamma)
    return (regularization_lambda * bias) / (regularization_lambda * len(sparse_user[user_id]) + regularization_gamma)

def update_movie_biases(movie_id, sparse_movie, user_latent_factors, movie_latent_factors, user_biases, regularization_lambda, regularization_gamma):
    """
    Update movie biases using stochastic gradient descent for matrix factorization.

    Args:
        movie_id (int): Movie index for which biases are updated.
        sparse_movie (list of lists): Sparse movie-user matrix.
        user_latent_factors (numpy.ndarray): User latent factors.
        movie_latent_factors (numpy.ndarray): Movie latent factors.
        user_biases (numpy.ndarray): User bias vector.
        regularization_lambda (float): Regularization parameter for model parameters.
        regularization_gamma (float): Regularization parameter for biases.

    Returns:
        float: Updated movie bias.
    """
    bias = 0

    for user_id, rating in sparse_movie[movie_id]:
        bias += rating - (np.dot(movie_latent_factors[movie_id], user_latent_factors[user_id]) + user_biases[user_id])

    #movie_biases[movie_id] = (regularization_lambda * bias) / (regularization_lambda * len(sparse_movie[movie_id]) + regularization_gamma)
    return (regularization_lambda * bias) / (regularization_lambda * len(sparse_movie[movie_id]) + regularization_gamma)

def calculate_loss(sparse_user, user_latent_factors, movie_latent_factors, user_biases, movie_biases, regularization_lambda, regularization_tau, regularization_gamma):
    """
    Calculate the loss function for matrix factorization with biases.

    Args:
        sparse_user (list of lists): Sparse user-movie matrix.
        user_latent_factors (numpy.ndarray): User latent factors.
        movie_latent_factors (numpy.ndarray): Movie latent factors.
        user_biases (numpy.ndarray): User bias vector.
        movie_biases (numpy.ndarray): Movie bias vector.
        regularization_lambda (float): Regularization parameter for model parameters.
        regularization_tau (float): Regularization parameter for user and movie latent factors.
        regularization_gamma (float): Regularization parameter for user and movie biases.

    Returns:
        float: Calculated loss value.
    """
    loss = 0

    for user_id in range(len(sparse_user)):
        for movie_id, rating in sparse_user[user_id]:
            loss += (rating - (np.dot(user_latent_factors[user_id], movie_latent_factors[movie_id]) + user_biases[user_id] + movie_biases[movie_id]))**2

    reg_user = 0

    for user_id in range(len(sparse_user)):
        reg_user += np.dot(user_latent_factors[user_id], user_latent_factors[user_id])

    reg_movie = 0

    for movie_id in range(len(movie_latent_factors)):
        reg_movie += np.dot(movie_latent_factors[movie_id], movie_latent_factors[movie_id])

    return (regularization_lambda / 2) * loss + (regularization_tau / 2) * reg_user + (regularization_tau / 2) * reg_movie + (regularization_gamma / 2) * np.dot(user_biases, user_biases) + (regularization_gamma / 2) * np.dot(movie_biases, movie_biases)

def calculate_rmse(sparse_user, user_latent_factors, movie_latent_factors, user_biases, movie_biases):
    """
    Calculate the Root Mean Squared Error (RMSE) for matrix factorization with biases.

    Args:
        sparse_user (list of lists): Sparse user-movie matrix.
        user_latent_factors (numpy.ndarray): User latent factors.
        movie_latent_factors (numpy.ndarray): Movie latent factors.
        user_biases (numpy.ndarray): User bias vector.
        movie_biases (numpy.ndarray): Movie bias vector.

    Returns:
        float: Calculated RMSE value.
    """
    error_sum = 0
    num_samples = 0

    for user_id in range(len(sparse_user)):
        for movie_id, rating in sparse_user[user_id]:
            error_sum += (rating - (np.dot(user_latent_factors[user_id], movie_latent_factors[movie_id]) + user_biases[user_id] + movie_biases[movie_id])) ** 2
            num_samples += 1

    rmse = np.sqrt(error_sum / num_samples)
    return rmse

def plot_loss_and_rmse(iteration, loss, rmse, savefig):
    """
    Plot the loss and RMSE values over iterations.

    Args:
        iteration (list): List of iteration numbers.
        loss (list): List of loss values over iterations.
        rmse (list): List of RMSE values over iterations.
        savefig (string): Name of the files in which we save the plots.

    Returns:
        None
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(iteration, loss)
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(iteration, rmse)
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")


    plt.savefig(savefig)
    plt.tight_layout()
    plt.show()

# Main function to train the model
def train_model(user_train, movie_train, user_test, movie_test, num_iterations, user_latent_factors, movie_latent_factors, user_biases, movie_biases, regularization_lambda, regularization_tau, regularization_gamma, latent_dimension):
    """
    Train the matrix factorization model.

    Args:
        user_train (list): Training data for users.
        movie_train (list): Training data for movies.
        user_test (list): Testing data for users.
        movie_test (list): Testing data for movies.
        num_iterations (int): Number of iterations for training.
        user_latent_factors (list): Latent factors for users.
        movie_latent_factors (list): Latent factors for movies.
        user_biases (list): User biases.
        movie_biases (list): Movie biases.
        regularization_lambda (float): Lambda regularization parameter.
        regularization_tau (float): Tau regularization parameter.
        regularization_gamma (float): Gamma regularization parameter.
        latent_dimension (int): Dimension of latent factors.

    Returns:
        iteration (list): List of iteration numbers.
        loss_train (list): List of training loss values over iterations.
        rmse_train (list): List of training RMSE values over iterations.
        loss_test (list): List of testing loss values over iterations.
        rmse_test (list): List of testing RMSE values over iterations.
    """
    iteration = []
    loss_train = []
    rmse_train = []

    loss_test = []
    rmse_test = []

    # Utilize tqdm to display the progress bar
    for i in tqdm(range(num_iterations)):
        for user_id in range(len(user_train)):
            user_biases[user_id] = update_user_biases(user_id, user_train, user_latent_factors, movie_latent_factors, movie_biases, regularization_lambda, regularization_gamma)

        for user_id in range(len(user_train)):
            user_latent_factors[user_id] = update_user_latent_factors(user_id, user_train, user_biases, movie_biases, movie_latent_factors, regularization_lambda, regularization_tau, latent_dimension)

        for movie_id in range(len(movie_train)):
            movie_biases[movie_id] = update_movie_biases(movie_id, movie_train, user_latent_factors, movie_latent_factors, user_biases, regularization_lambda, regularization_gamma)

        for movie_id in range(len(movie_train)):
            movie_latent_factors[movie_id] = update_movie_latent_factors(movie_id, movie_train, user_biases, movie_biases, user_latent_factors, regularization_lambda, regularization_tau, latent_dimension)


        iteration.append(i)
        loss_train.append(calculate_loss(user_train, user_latent_factors, movie_latent_factors, user_biases, movie_biases, regularization_lambda, regularization_tau, regularization_gamma))
        rmse_train.append(calculate_rmse(user_train, user_latent_factors, movie_latent_factors, user_biases, movie_biases))
        #print("Iteration ", i, " Loss train = ", loss_train[i])
        #print("Iteration ", i, " RMSE train = ", loss_train[i])

        loss_test.append(calculate_loss(user_test, user_latent_factors, movie_latent_factors, user_biases, movie_biases, regularization_lambda, regularization_tau, regularization_gamma))
        rmse_test.append(calculate_rmse(user_test, user_latent_factors, movie_latent_factors, user_biases, movie_biases))
        #print("Iteration ", i, " Loss test = ", loss_test[i])
        #print("Iteration ", i, " RMSE test = ", rmse_test[i])

    # plot_loss_and_rmse(iteration, losses, rmses)
    return iteration, loss_train, rmse_train, loss_test, rmse_test

iteration, loss_train, rmse_train, loss_test, rmse_test = train_model(user_train, movie_train, user_test, movie_test, num_iterations, user_latent_factors, movie_latent_factors, user_biases, movie_biases, regularization_lambda, regularization_tau, regularization_gamma, latent_dimension)

print("RMSE train for each iteration:")
print(rmse_train)

print("RMSE test for each iteration:")
print(rmse_test)

print("Loss train for each iteration:")
print(loss_train)

print("Loss test for each iteration:")
print(loss_test)

plot_loss_and_rmse(iteration, loss_train, rmse_train, 'loss_RMSE_train_2.jpg')

plot_loss_and_rmse(iteration, loss_test, rmse_test, 'loss_RMSE_test_2.jpg')

def plot_rmse_test_train(iteration, rmse_train, rmse_test):
    """
    Plot RMSE values for training and testing datasets over iterations.

    Args:
        iteration (list): List of iteration numbers.
        rmse_train (list): List of training RMSE values.
        rmse_test (list): List of testing RMSE values.
    """
    plt.plot(iteration, rmse_train, label="Train RMSE")
    plt.plot(iteration, rmse_test, label="Test RMSE")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.legend()

    plt.tight_layout()
    plt.grid()
    plt.savefig('RMSE_test_train_2.pdf')
    plt.show()

plot_rmse_test_train(iteration, rmse_train, rmse_test)

# Define a dictionary to store the data you want to save
saved_data = {
    'user_latent_factors': user_latent_factors,
    'movie_latent_factors': movie_latent_factors,
    'user_biases': user_biases,
    'movie_biases': movie_biases,
    'iteration': iteration,
    'loss_train': loss_train,
    'rmse_train': rmse_train,
    'loss_test': loss_test,
    'rmse_test': rmse_test
}

# Specify the file path where you want to save the data
save_path = 'saved_data_16.pkl'

# Use Pickle to serialize and save the data
with open(save_path, 'wb') as file:
    pickle.dump(saved_data, file)

print("Data saved successfully.")





