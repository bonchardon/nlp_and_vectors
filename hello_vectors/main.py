import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import w3_unittest

from utils import get_vectors

data = pd.read_csv('data/capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']

# print first five elements in the DataFrame
print(data.head(5))

word_embeddings = pickle.load(open("./data/word_embeddings_subset.p", "rb"))
# print(word_embeddings)
#
# print("dimension: {}".format(word_embeddings['Ukraine'].shape[0]))
#
# print(len(word_embeddings['Turkey']))


# 1: cosine similarity

def cosine_similarity(A, B):

    dot = np.dot(A, B)
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)

    cos = dot / (normA * normB)

    return cos


def euclidean_distance(A, B):

    d = np.linalg.norm(A - B)
    return d


def get_country(city1, country1, city2, embeddings):
    """
        Input:
            city1: a string (the capital city of country1)
            country1: a string (the country of capital1)
            city2: a string (the capital city of country2)
            embeddings: a dictionary where the keys are words and values are their emmbeddings
        Output:
            countries: a dictionary with the most likely country and its similarity score
    """
    group = set((city1, country1, city2))
    city1_embed = word_embeddings[city1]
    country1_embed = word_embeddings[country1]
    city2_embed = word_embeddings[city2]

    # getting embeddings for country2
    vec = country1_embed - city1_embed + city2_embed

    # Initialize the similarity to -1 (it will be replaced by a similarities that are closer to +1)
    similarity = -1

    # initialize country to an empty string
    country = ''

    for word in embeddings.keys():
        # if the word at iteration i not in our group
        if word not in group:
            word_embedding = embeddings[word]
            # caculate the similarity between the word and our vector
            cur_similarity = cosine_similarity(vec, word_embedding)

            if cur_similarity > similarity:
                # update similarity to search for a better one
                similarity = cur_similarity

                country = (word, similarity)

    return country


def get_accuracy(word_embeddings, data):

    '''
    Accuracy:

            correct # of predictions / total # of predictions

    :param word_embeddings:
    :param data:
    :return: accuracy
    '''

    # initialize number of correct predictions to 0
    correct_prediction = 0

    print(data.head(4))

    for i, row in data.iterrows():
        city1 = row['city1']
        country1 = row['country1']
        city2 = row['city2']
        country2 = row['country2']

        # use get_country to find the predicted country2
        predicted_country2, _ = get_country(city1, country1, city2, word_embeddings)

        # if the predicted country2 is the same as the actual country2...
        if predicted_country2 == country2:
            # increment the number of correct by 1
            correct_prediction += 1

    # get the number of rows in the data dataframe (length of dataframe)
    m = len(data)

    # calculate the accuracy by dividing the number correct by m
    accuracy = correct_prediction / m

    return accuracy


def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    # mean center the data
    X_demeaned = X - np.mean(X, axis=0)

    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned, rowvar=False)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix, UPLO='L')

    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals)

    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[::-1]

    # sort the eigen values by idx_sorted_decreasing
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or n_components)
    eigen_vecs_subset = eigen_vecs_sorted[:, 0:n_components]

    # transform the data by multiplying the transpose of the eigenvectors with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced = np.dot(eigen_vecs_subset.T, X_demeaned.T).T

    return X_reduced


if __name__ == "__main__":
    word_one = word_embeddings['Paris']
    word_two = word_embeddings['Ukraine']
    word_three = word_embeddings['France']

    cos_sim = cosine_similarity(word_one, word_two)
    print('Cosine similarity (1 and 2) ==> ', cos_sim)

    cos_sim = cosine_similarity(word_one, word_three)
    print('Cosine similarity (1 and 3) ==> ', cos_sim)

    euclid_dist = euclidean_distance(word_one, word_two)
    print('Euclidean distance (1 and 2) ==> ', euclid_dist)

    euclid_dist = euclidean_distance(word_one, word_three)
    print('Euclidean distance (1 and 3) ==> ', euclid_dist)

    evaluation_data = []

    try:
        country = get_country('Kiev', 'Ukraine', 'Paris', word_embeddings)
        print(country)
    except KeyError as e:
        print(f'Unfortunately, there is a mistake! I assume this city either nonexistent, '
              f'or who knows what. The mistake is ==> {e}!')

    accuracy = get_accuracy(word_embeddings, data)
    print(f"Accuracy is {accuracy}")

    # np.random.seed(1)
    # X = np.random.rand(3, 10)
    # X_reduced = compute_pca(X, n_components=2)
    # print("Your original matrix was " + str(X.shape) + " and it became:")
    # print(X_reduced)

    """
    Visualization part ==> 
    """

    words = ['Paris', 'France', 'Kiev', 'Ukraine', 'gas', 'happy', 'sad', 'city', 'town',
             'village', 'country', 'continent', 'petroleum', 'joyful']

    # given a list of words and the embeddings, it returns a matrix with all the embeddings
    X = get_vectors(word_embeddings, words)

    print(f'You have {len(words)} words each of 300 dimensions thus X.shape is:', X.shape)

    # We have done the plotting for you. Just run this cell.
    result = compute_pca(X, 2)
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

    plt.show()
