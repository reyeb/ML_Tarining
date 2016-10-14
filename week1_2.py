import sframe
import pandas
import os
import math
import scipy
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from utils.listhelper import *
from sklearn import linear_model
from sklearn.metrics import accuracy_score

# Q 16 onward


def remove_punctuation(text):
    import string
    # replace punctuations in a test (translate string.punctuation) with None
    return text.translate(None, string.punctuation)


def Read_and_Prepare_data():
    products = sframe.SFrame('amazon_baby.gl/').dropna()
    products = products.fillna('review','')  # fill in N/A's in the review c# olumn
    products['review_clean'] = products['review'].apply(remove_punctuation)

    products = products[products['rating'] != 3]
    # Now, we will assign reviews with a rating of 4 or higher to be positive reviews, while the ones with rating of 2 or lower are negative.
    products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
    # Let's perform a train/test split with 80% of the data in the training set and 20% of the data in the test set. If you are using SFrame, make sure to use seed=1 so that you get the same result as everyone else does. (
    train_data, test_data = products.random_split(.8, seed=1)
    return train_data,test_data



def build_a_bag_of_words_data(train_data,test_data):
    # see http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
    # .sparse scipy : to deal with alot of zeros in a vector like bagsofwords
    # o, I'm doing some Kmeans classification using numpy arrays that are quite sparse-- lots and lots of zeroes. I figured that I'd use scipy's 'sparse' package to reduce the storage overhead
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

    # Use this token pattern to keep single-letter words
    # First, learn vocabulary from the training data and assign columns to words
    # Then convert the training data into a sparse matrix
    train_matrix = vectorizer.fit_transform(train_data['review_clean'])
    # get the feature names
    feature_names= vectorizer.get_feature_names()
    # Second, convert the test data into a sparse matrix, using the same word-column mapping
    test_matrix = vectorizer.transform(test_data['review_clean'])
    return train_matrix,test_matrix,feature_names

def build_a_bag_of_words_data_significant_words(train_data,test_data,significant_words):
    # see http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
    # .sparse scipy : to deal with alot of zeros in a vector like bagsofwords
    # o, I'm doing some Kmeans classification using numpy arrays that are quite sparse-- lots and lots of zeroes. I figured that I'd use scipy's 'sparse' package to reduce the storage overhead
    vectorizer = CountVectorizer(vocabulary=significant_words) # limit to 20 words
    feature_names= vectorizer.vocabulary
    # Use this token pattern to keep single-letter words
    # First, learn vocabulary from the training data and assign columns to words
    # Then convert the training data into a sparse matrix
    train_matrix = vectorizer.fit_transform(train_data['review_clean'])
    # Second, convert the test data into a sparse matrix, using the same word-column mapping
    test_matrix = vectorizer.transform(test_data['review_clean'])
    return train_matrix,test_matrix,feature_names

def Build_a_model_using_train_data(train_matrix,train_data):
    sentiment_model = linear_model.LogisticRegression()
    sentiment_model.fit(train_matrix, train_data['sentiment'])
    return sentiment_model

def make_class_based_prediction(sentiment_model,test_matrix):
    scores = sentiment_model.decision_function(test_matrix)
    prediction=  sentiment_model.predict(test_matrix)
    print "Prediction",prediction

    # calculate probabilities manually by changing score to probabilities
    for s in scores:
        prob=1. / (1. + np.exp(-1*s))
        print s,prob

    return prediction

def make_prediction_with_probability_function(sentiment_model,test_matrix):
    prediction = sentiment_model.predict_proba(test_matrix)
    probabilities = prediction[:, 1]
    list_of_all_probabilities = list(probabilities)

    return prediction,probabilities

def get_top_n_high_probability_results(N,probabilities,test_data):
    # arr_probabilities = np.array(probabilities)
    # Index_of_sorted_list_arr = arr_probabilities.argsort()[-20:][::-1]
    Index_of_sorted_list_arr=listhelper().get_index_of_top_n_data(probabilities,20)
    Index_of_sorted_list = list(Index_of_sorted_list_arr)
    for p in Index_of_sorted_list:
        print test_data[p]

# see this:http://blog.revolutionanalytics.com/2016/03/classification-models.html
# Used to check if our prediction is better than assigning all the prediction the same class of the major class in
# the data
def majority_class_classifier(actual):
    num_positive = ( actual == +1).sum()
    num_negative = ( actual== -1).sum()
    if num_positive>num_negative:
        majority_class=+1
    else:
        majority_class=-1

    # after getting wjhat the major class is
    print num_positive
    print num_negative

    # build and array where all the prediction is the major class: e.g. a list of all +1
    class_prediction_array=[majority_class]*len(actual)
    print class_prediction_array
    # use this to calculate the accuracy
    acc_majority=calculate_accuracy(actual,class_prediction_array)

    print "Majority_class_classifier_accuracy _is:",acc_majority

def calculate_accuracy(actual,prediction):
    # print "actual data:",test_data['sentiment']
    actual_data_array=np.array(actual)
    prediction_array=np.array(prediction)
    accuracy= accuracy_score(actual_data_array, prediction_array)
    return accuracy
    # calcualte by hand
    # correct= np.sum(prediction==actual_data_array)
    # print correct/float(actual_data_array.size)


# Consider all the words and make the fitting and prediction
def general_prediction_pipeline():

    train_data,test_data=Read_and_Prepare_data()
    train_matrix,test_matrix,feature_names=build_a_bag_of_words_data(train_data,test_data)
    sentiment_model=Build_a_model_using_train_data(train_matrix,train_data)
    positive_coef = sentiment_model.coef_[sentiment_model.coef_ >= 0]
    print "len of positive coefficients", len(positive_coef)

    coef_table = sframe.SFrame({'word': feature_names,
                                'coefficient': sentiment_model.coef_.flatten()})
    #print the top n coefficient; their names and their weights
    print coef_table.topk('coefficient', k=20)
    positive_coef = coef_table[coef_table['coefficient'] >= 0]
    print positive_coef
    #
    # prediction,probabilities=make_prediction_with_probability_function(sentiment_model,test_matrix)
    # get_top_n_high_probability_results(20,probabilities,test_data)

# Focuse on specific words when buildng the classifier
def specific():
    significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
          'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
          'work', 'product', 'money', 'would', 'return']
    train_data,test_data=Read_and_Prepare_data()
    train_matrix,test_matrix,feature_names=build_a_bag_of_words_data_significant_words(train_data,test_data,significant_words)

    simple_model = Build_a_model_using_train_data(train_matrix, train_data)

    # print simple_model.get_params()

    coef_table = sframe.SFrame({'word': significant_words,
                                'coefficient': simple_model.coef_.flatten()})
    print coef_table
    positive_coef = coef_table[coef_table['coefficient'] >= 0]
    print positive_coef

    prediction = simple_model.predict(test_matrix)

    actual_classification=test_data['sentiment']
    print calculate_accuracy(actual_classification, prediction)

    majority_class_classifier(train_data['sentiment'])
    majority_class_classifier(test_data['sentiment'])

specific()
