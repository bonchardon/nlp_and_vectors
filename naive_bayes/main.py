from utils import process_tweet, lookup
import pdb
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd
# import w2_unittest


nltk.download('twitter_samples')
nltk.download('stopwords')

filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)


def count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = word, y

            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1

    return result


def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels corresponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0

    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    # print(vocab)
    V = len(vocab)

    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]

    # Calculate D, the number of documents
    D = len(train_y)

    # Calculate D_pos, the number of positive documents
    D_pos = np.sum(train_y)

    # Calculate D_neg, the number of negative documents
    D_neg = D - D_pos

    # Calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood


def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    return p


def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    """
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    accuracy = 0  # Initialize accuracy

    y_hats = []  # Initialize a list to store predicted labels

    for tweet in test_x:
        # Predict the sentiment for each tweet using naive_bayes_predict
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1  # Predicted class is 1
        else:
            y_hat_i = 0  # Predicted class is 0

        y_hats.append(y_hat_i)  # Append the predicted class to the list y_hats

    # Calculate the error as the average of the absolute differences between y_hats and test_y
    error = np.mean(np.absolute(y_hats-test_y))
    # Calculate the accuracy as 1 minus the error
    accuracy = 1 - error

    return accuracy


def get_ratio(freqs, word):
    '''
    Input:
        freqs: dictionary containing the words

    Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
    '''
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    # use lookup() to find positive counts for the word (denoted by the integer 1)
    pos_neg_ratio['positive'] = lookup(freqs, word, 1)

    # use lookup() to find negative counts for the word (denoted by integer 0)
    pos_neg_ratio['negative'] = lookup(freqs, word, 0)

    # calculate the ratio of positive to negative counts for the word
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1) / (pos_neg_ratio['negative'] + 1)
    return pos_neg_ratio


def get_words_by_threshold(freqs, label, threshold, get_ratio=get_ratio):
    '''
    Input:
        freqs: dictionary of words
        label: 1 for positive, 0 for negative
        threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
    Output:
        word_list: dictionary containing the word and information on its positive count, negative count, and ratio of positive to negative counts.
        example of a key value pair:
        {'happi':
            {'positive': 10, 'negative': 20, 'ratio': 0.5}
        }
    '''
    word_list = {}

    for key in freqs.keys():
        word, _ = key

        # get the positive/negative ratio for a word
        pos_neg_ratio = get_ratio(freqs, word)

        # if the label is 1 and the ratio is greater than or equal to the threshold...
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:

            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio

        # If the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:

            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio

        # otherwise, do not include this word in the list (do nothing)

    return word_list


if __name__ == "__main__":
    # get the sets of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    # split the data into two pieces, one for training and one for testing (validation set)
    test_pos = all_positive_tweets[4000:]
    train_pos = all_positive_tweets[:4000]
    test_neg = all_negative_tweets[4000:]
    train_neg = all_negative_tweets[:4000]

    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

    # avoid assumptions about the length of all_positive_tweets
    train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
    test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

    result = {}
    tweets = ['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
    ys = [1, 0, 0, 0, 0]
    freqs = count_tweets({}, train_x, train_y)
    print(count_tweets)

    logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
    print(logprior)
    print(len(loglikelihood))

    # testing part!
    my_tweet = 'She smiled.'
    p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
    print('The expected output is', p)

    my_tweet = 'She died.'
    p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
    print('The expected output is', p)

    # accuracy
    print("Naive Bayes accuracy = %0.4f" %
          (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

    for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great',
                  'great great great', 'great great great great']:
        # print( '%s -> %f' % (tweet, naive_bayes_predict(tweet, logprior, loglikelihood)))
        p = naive_bayes_predict(tweet, logprior, loglikelihood)
        #     print(f'{tweet} -> {p:.2f} ({p_category})')
        print(f'{tweet} -> {p:.2f}')

    my_tweet1 = 'you are bad :('
    naive_bayes_predict1 = naive_bayes_predict(my_tweet1, logprior, loglikelihood)
    print(naive_bayes_predict1)

    my_tweet2 = 'you are good!'
    naive_bayes_predict2 = naive_bayes_predict(my_tweet2, logprior, loglikelihood)
    print(naive_bayes_predict2)

    print(get_ratio(freqs, 'happi'))

    print(get_words_by_threshold(freqs, label=0, threshold=0.05))

    print('Truth Predicted Tweet')
    for x, y in zip(test_x, test_y):
        y_hat = naive_bayes_predict(x, logprior, loglikelihood)
        if y != (np.sign(y_hat) > 0):
            print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(
                process_tweet(x)).encode('ascii', 'ignore')))

    # FINAL CHECK
    my_tweet_happy = 'I am happy because I am learning'
    my_tweet_sad = 'Iam sad because I am learning'

    p_happy = naive_bayes_predict(my_tweet_happy, logprior, loglikelihood)
    print(p_happy)

    p_sad = naive_bayes_predict(my_tweet_sad, logprior, loglikelihood)
    print(p_sad)



