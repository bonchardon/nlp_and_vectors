import pdb
import pickle
import string

import time

import nltk
import numpy as np
from nltk.corpus import stopwords, twitter_samples

from utils import (cosine_similarity, get_dict,
                   process_tweet)
from os import getcwd


def get_matrices(en_fr, french_vecs, english_vecs):
    """
    Input:
        en_fr: English to French dictionary
        french_vecs: French words to their corresponding word embeddings.
        english_vecs: English words to their corresponding word embeddings.
    Output:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the projection matrix that minimizes the F norm ||X R -Y||^2.
    """

    # X_l and Y_l are lists of the english and french word embeddings
    X_l = list()
    Y_l = list()

    # get the english words (the keys in the dictionary) and store in a set()
    english_set = english_vecs.keys()

    # get the french words (keys in the dictionary) and store in a set()
    french_set = french_vecs.keys()

    # store the french words that are part of the english-french dictionary (these are the values of the dictionary)
    french_words = set(en_fr.values())

    # loop through all english, french word pairs in the english french dictionary
    for en_word, fr_word in en_fr.items():

        # check that the french word has an embedding and that the english word has an embedding
        if fr_word in french_set and en_word in english_set:

            # get the english embedding
            en_vec = english_vecs[en_word]

            # get the french embedding
            fr_vec = french_vecs[fr_word]

            # add the english embedding to the list
            X_l.append(en_vec)

            # add the french embedding to the list
            Y_l.append(fr_vec)

    # stack the vectors of X_l into a matrix X
    X = np.vstack(X_l)

    # stack the vectors of Y_l into a matrix Y
    Y = np.vstack(Y_l)

    return X, Y


def compute_loss(X, Y, R):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
    '''
    # m is the number of rows in X
    m = X.shape[0]

    # diff is XR - Y
    diff = np.dot(X, R) - Y

    # diff_squared is the element-wise square of the difference
    diff_squared = diff ** 2

    # sum_diff_squared is the sum of the squared elements
    sum_diff_squared = np.sum(diff_squared)

    # loss i is the sum_diff_squared divided by the number of examples (m)
    loss = sum_diff_squared / m
    return loss


def compute_gradient(X, Y, R):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        g: a scalar value - gradient of the loss function L for given X, Y and R.
    '''
    # m is the number of rows in X
    m = X.shape[0]

    # gradient is X^T(XR - Y) * 2/m
    gradient = np.dot(X.T, np.dot(X, R) - Y) * (2 / m)

    return gradient


def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003, verbose=True, compute_loss=compute_loss, compute_gradient=compute_gradient):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        train_steps: positive int - describes how many steps will gradient descent algorithm do.
        learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
    Outputs:
        R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
    '''
    np.random.seed(129)

    # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
    # R is a square matrix with length equal to the number of dimensions in th  word embedding
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(train_steps):
        if verbose and i % 25 == 0:
            print(f"loss at iteration {i} is: {compute_loss(X, Y, R):.4f}")
        ### START CODE HERE ###
        # use the function that you defined to compute the gradient
        gradient = compute_gradient(X, Y, R)

        # update R by subtracting the learning rate times gradient
        R -= learning_rate * gradient
        ### END CODE HERE ###
    return R


# UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def nearest_neighbor(v, candidates, k=1, cosine_similarity=cosine_similarity):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    """
    similarity_l = []

    # for each candidate vector...
    for row in candidates:
        # get the cosine similarity
        cos_similarity = cosine_similarity(v, row)

        # append the similarity to the list
        similarity_l.append(cos_similarity)

    # sort the similarity list and get the indices of the sorted list
    # sorted_ids = np.argsort(similarity_l)
    sorted_ids = np.sort(similarity_l)

    # Reverse the order of the sorted_ids array
    sorted_ids = sorted_ids[::-1]

    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[-k:]

    return k_idx


if __name__ == "__main__":
    # add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path
    filePath = f"{getcwd()}/tmp2/"
    nltk.data.path.append(filePath)

    en_embeddings_subset = pickle.load(open("./data/en_embeddings.p", "rb"))
    fr_embeddings_subset = pickle.load(open("./data/fr_embeddings.p", "rb"))

    # loading the english to french dictionaries
    en_fr_train = get_dict('./data/en-fr.train.txt')
    # print('The length of the English to French training dictionary is', len(en_fr_train))  # 5000
    en_fr_test = get_dict('./data/en-fr.test.txt')
    # print('The length of the English to French test dictionary is', len(en_fr_test))  # 1500

    # get_matrices (1)
    X_train, Y_train = get_matrices(
        en_fr_train, fr_embeddings_subset, en_embeddings_subset)
    # print(X_train, Y_train)

    # compute_loss (2)
    np.random.seed(123)
    m = 10
    n = 5
    X = np.random.rand(m, n)
    Y = np.random.rand(m, n) * .1
    R = np.random.rand(n, n)
    print(f"Expected loss for an experiment with random matrices: {compute_loss(X, Y, R):.4f}")

    # compute_gradient (3)
    np.random.seed(123)
    m = 10
    n = 5
    X = np.random.rand(m, n)
    Y = np.random.rand(m, n) * .1
    R = np.random.rand(n, n)
    gradient = compute_gradient(X, Y, R)
    print(f"First row of the gradient matrix: {gradient[0]}")

    # align embeddings (4)
    np.random.seed(129)
    m = 10
    n = 5
    X = np.random.rand(m, n)
    Y = np.random.rand(m, n) * .1
    R = align_embeddings(X, Y)

    # R_train = align_embeddings(X_train, Y_train, train_steps=400, learning_rate=0.8)

    # nearest neighbour (5)
    # Test your implementation:
    v = np.array([1, 0, 1])
    candidates = np.array([[1, 0, 5], [-2, 5, 3], [2, 0, 1], [6, -9, 5], [9, 9, 9]])
    print(candidates[nearest_neighbor(v, candidates, 3)])
