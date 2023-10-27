#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is for loading data, doing recommendation and ploting the 2D embedding item

@author: fandresena
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

# Specify the path to the saved data file
load_path = 'data_begin.pkl'

# Load the saved data using Pickle
with open(load_path, 'rb') as file:
    loaded_data = pickle.load(file)

# Access the values from the loaded data
map_user_index = loaded_data['map_user_index']
map_index_user = loaded_data['map_index_user']
map_movie_index = loaded_data['map_movie_index']
map_index_movie = loaded_data['map_index_movie']
sparse_user = loaded_data['sparse_user']
sparse_movie = loaded_data['sparse_movie']
user_train = loaded_data['user_train']
movie_train = loaded_data['movie_train']
user_test = loaded_data['user_test']
movie_test = loaded_data['movie_test']

load_path_bias = 'bias-only_data.pkl'

# Load the saved data using Pickle
with open(load_path_bias, 'rb') as file:
    loaded_data_bias = pickle.load(file)

# Access the values from the loaded data
iteration_numbers = loaded_data_bias['iteration']
loss_train_2 = loaded_data_bias['loss_train']
rmse_train_2 = loaded_data_bias['rmse_train']
loss_test_2 = loaded_data_bias['loss_test']
rmse_test_2 = loaded_data_bias['rmse_test']

# Specify the path to the saved data file
load_path = 'saved_data_16.pkl'

# Load the saved data using Pickle
with open(load_path, 'rb') as file:
    loaded_data = pickle.load(file)

# Access the values from the loaded data
user_latent_factors = loaded_data['user_latent_factors']
movie_latent_factors = loaded_data['movie_latent_factors']
user_biases = loaded_data['user_biases']
movie_biases = loaded_data['movie_biases']
iteration = loaded_data['iteration']
loss_train = loaded_data['loss_train']
rmse_train = loaded_data['rmse_train']
loss_test = loaded_data['loss_test']
rmse_test = loaded_data['rmse_test']

def get_map(name,data):
    #Retrieve the index of a user or movie by name from the mapping dictionary.
    return data[name]

def check(user,movie,user_mapping,movie_mapping):
    #Make a rating prediction for a user-movie pair.
    user_index=get_map(user,user_mapping)
    movie_index=get_map(movie,movie_mapping)
    pred=user_latent_factors[user_index] @ movie_latent_factors[movie_index] + user_biases[user_index] + movie_biases[movie_index]
    return pred

def precision(map_user_index, map_movie_index, user_test, difference = 1.0, rating_test = 5.0):
    """
    Calculate the precision of a recommendation system.

    Args:
        map_user_index (dict): Maps userId to user index.
        map_movie_index (dict): Maps movieId to movie index.
        user_test (list of lists): Testing data for users.
        difference (float): Maximum difference between predicted and actual ratings (default is 1.0).
        rating_test (float): Threshold rating to consider (default is 5.0).

    Returns:
        float: Precision as a percentage.
    """
    count_precision = 0
    count_rating = 0
    for i in range(len(user_test)):
        user = user_test[i]
        user_name = map_index_user[i]
        for movie, rating in user:
            if rating <= rating_test:
                movie_name = map_index_movie[movie]
                prediction = check(user_name, movie_name, map_user_index, map_movie_index)
                if abs(prediction - rating) >= difference:
                    count_precision +=1
                count_rating += 1
    precision = count_precision / count_rating
    return precision * 100

precision(map_user_index, map_movie_index, user_test, 0.01, 5.0)

def read_csv_as_dict_list(csv_file):
    """
    Read data from a CSV file and store it as a list of dictionaries.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        list of dict: List of dictionaries representing the data from the CSV file.
    """
    data = []

    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data.append(row)
        return data
    except FileNotFoundError:
        print(f"File '{csv_file}' not found.")
        return None

# Usage: Read CSV data into a NumPy array of dictionaries
csv_file = 'movies.csv'
movie_array = np.array(read_csv_as_dict_list(csv_file), dtype=object)

##################################################################################
####For the following function, the name of movie have to be in lower case!!!!####
##################################################################################
def search_movies(movie_array, search_term):
    """
    Search for movies in the data based on a search term.

    Args:
        movie_array (list of dict): Data containing movie information.
        search_term (str): Term to search for in movie titles.

    Returns:
        list: List of movie IDs as strings that match the search term.
    """
    matching_movies = [movie for movie in movie_array if search_term in movie['title'].lower()]

    lord_array = np.array([])

    # Print the matching movies
    if matching_movies:
        print(f"Movies containing '{search_term}':")
        for movie in matching_movies:
            print(f"{movie['movieId']} {movie['title']}, {movie['genres']}")
            lord_array = np.append(lord_array, str(movie['movieId']))
    else:
        print(f"No matching movies found for '{search_term}'.")

    # Convert movie IDs to strings with '.0'
    lord_array = np.array([s + '.0' for s in lord_array])
    return lord_array

def get_index(string_index):
    """
    Get the index corresponding to a string index from the map_movie_index dictionary.

    Args:
        string_index (str): A string index to look up in the map_movie_index dictionary.

    Returns:
        int: The integer index corresponding to the provided string index.
    """
    index = map_movie_index[string_index]
    return index

def get_movie(index):
    """
    Get the movie index (with optional '.0' suffix removed) corresponding to the provided index.

    Args:
        index (int or str): The index or string index for a movie.

    Returns:
        str: The movie index with the '.0' suffix removed (if present).
    """
    movie_index = map_index_movie[index]
    movie_index = movie_index.replace('.0', '')
    return movie_index


def find_movie_title(data, movie_id):
    """
    Find the title of a movie based on its movie ID in the given dataset.

    Args:
        data (list of dict): A dataset containing movie information.
        movie_id (int): The movie ID to search for.

    Returns:
        str or None: The title of the movie with the specified ID, or None if not found.
    """
    for movie in data:
        if movie['movieId'] == movie_id:
            return movie['title']
    return None

def recommend_movies(movie_latent_factors, movie_biases, user_index, num_recommendations=10):
    '''
    Generate movie recommendations for a user based on latent factors and biases.

    Args:
        movie_latent_factors (np.array): Latent factors for movies.
        movie_biases (np.array): Biases for movies.
        user_index (int): Index of the user for whom recommendations are generated.
        num_recommendations (int): Number of movie recommendations to return (default is 10).

    Returns:
        list of tuples: List of (predicted_rating, movie_index) pairs for recommended movies.
    '''
    dummy_user = movie_latent_factors[user_index].copy()
    recommender_array = [(np.dot(dummy_user.T, movie_latent_factors[i]) + 0.05 * movie_biases[i], i) for i in range(len(movie_latent_factors))]
    sorted_recommender = sorted(recommender_array, key=lambda x: x[0], reverse=True)

    return sorted_recommender[:num_recommendations]

sorted_recommender = recommend_movies(movie_latent_factors, movie_biases, get_index('4993.0'), num_recommendations=10)

def print_movie_info_for_top_recommended(movie_array, sorted_recommender, map_index_movie, top_count=5):
    # Define the headers
    headers = ["   movieId", "title", "genres"]
    recommendation = []

    # Iterate through the top recommended movies
    for i in range(top_count):
        movie = sorted_recommender[i][1]
        movieId = map_index_movie[movie]
        movieId = movieId.replace(".0", "")

        # Use a list comprehension to find the matching row
        matching_rows = [row for row in movie_array if row['movieId'] == movieId]
        recommendation.append(matching_rows)

        if matching_rows:
            data = [[f"   {row['movieId']}", row['title'], row['genres']] for row in matching_rows]
            if i > 0:
                # If it's not the first movie, add an empty line
                print()
            print(tabulate.tabulate(data, headers, tablefmt="grid"))
        else:
            print("No rows found with movieId '{}'.".format(movieId))
    return recommendation

# Usage example
print_movie_info_for_top_recommended(movie_array, sorted_recommender, map_index_movie, top_count=5)

# This function is used to plot the 2D embedding movie
def plot_scatter_and_quiver(X, Y, color='black', edgecolor='green'):
    # Create scatter plot
    plt.scatter(X, Y, c="pink", linewidths=2, marker="s", edgecolor=edgecolor, s=50)

    plt.xlim(-2, 2)
    plt.ylim(-2, 3)

#This code represent the lists of movies genres
hero = ['122914', '195159', '122920']
fantastic = ['7153', '98809', '4993', '106489']
kids = ['3114', '34', '1', '2384']
comedy = ['86377', '86347', '90522', '6550', '3869', '1665']
horror = ['48877', '39446', '8957', '6880', '5679']

#This function return list of movie index
def get_index_list(lists):
    index_list = []
    for i in range(len(lists)):
        listing = lists[i] + '.0'
        index_list.append(get_index(listing))
    return index_list

#The code starting from this line is for plotting the 2D embedding movie
hero = get_index_list(hero)
horror = get_index_list(horror)
fantastic = get_index_list(fantastic)
kids = get_index_list(kids)
comedy = get_index_list(comedy)
#thriller = get_index_list(thriller)

x_fantastic = [movie_latent_factors[i][13] for i in fantastic]
y_fantastic = [movie_latent_factors[i][2] for i in fantastic]
plot_scatter_and_quiver(x_fantastic, y_fantastic, color='black', edgecolor='green')

x_kids = [movie_latent_factors[i][13] for i in kids]
y_kids = [movie_latent_factors[i][2] for i in kids]
plot_scatter_and_quiver(x_kids, y_kids, color='blue', edgecolor='red')

x_horror = [movie_latent_factors[i][13] for i in horror]
y_horror = [movie_latent_factors[i][2] for i in horror]
plot_scatter_and_quiver(x_horror, y_horror, color='red', edgecolor='blue')

x_hero = [movie_latent_factors[i][13] for i in hero]
y_hero = [movie_latent_factors[i][2] for i in hero]
plot_scatter_and_quiver(x_hero, y_hero, color='purple', edgecolor='orange')

x_comedy = [movie_latent_factors[i][13] for i in comedy]
y_comedy = [movie_latent_factors[i][2] for i in comedy]
plot_scatter_and_quiver(x_comedy, y_comedy, color='yellow', edgecolor='brown')

x_horror = [movie_latent_factors[i][0] for i in horror]
x_horror

fig, ax = plt.subplots(figsize=(16, 10))

# Define an offset for the text labels
label_offset = -0.04  # Adjust this value as needed

# Scatter plot and label for 'fantastic'
ax.scatter(x_fantastic, y_fantastic, label='Fantastic')
for i in range(len(fantastic)):
    plt.text(x_fantastic[i] + label_offset, y_fantastic[i] + label_offset, find_movie_title(movie_array, get_movie(fantastic[i])), ha='center', va='center')

# Scatter plot and label for 'kids'
ax.scatter(x_kids, y_kids, label='Kids')
for i in range(len(kids)):
    plt.text(x_kids[i] + label_offset, y_kids[i] + label_offset, find_movie_title(movie_array, get_movie(kids[i])), ha='center', va='center')

# Scatter plot and label for 'horror'
ax.scatter(x_horror, y_horror, label='Horror')
for i in range(len(horror)):
    plt.text(x_horror[i] + label_offset, y_horror[i] + label_offset, find_movie_title(movie_array, get_movie(horror[i])), ha='center', va='center')

# Scatter plot and label for 'hero'
ax.scatter(x_hero, y_hero, label='Super-Hero movie')
for i in range(len(hero)):
    plt.text(x_hero[i] + label_offset, y_hero[i] + label_offset, find_movie_title(movie_array, get_movie(hero[i])), ha='center', va='center')

# Scatter plot and label for 'fantastic'
ax.scatter(x_comedy, y_comedy, label='Comedy')
for i in range(len(comedy)):
    plt.text(x_comedy[i] + label_offset, y_comedy[i] + label_offset, find_movie_title(movie_array, get_movie(comedy[i])), ha='center', va='center')

# Add a legend
ax.legend()

plt.savefig('item_embed.pdf')
plt.show()
