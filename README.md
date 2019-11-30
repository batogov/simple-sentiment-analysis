# Sentiment Analysis

Read the [Russian version](./README_RU.md) of the document 🇷🇺

## Problem description

Make a classifier of the emotional tonality of short text snippets (for example, tweets). The algorithm should be able to classify messages into two classes: messages with positive emotionality and messages with negative emotionality.

## Tools and libraries

* Python 3;
* Pandas for working with data;
* Scikit-Learn for tokenization, cross-validation and machine learning algorithms;
* PyMorphy2 for lemmatization of Russian words.

## Training set

Creating a training sample is a complex and time-consuming activity. It takes a lot of time and the help of a lot of people (assessors) to create and classify even a small training set.

I will use the [ready set](http://study.mokoron.com/) consisting of approximately 225 thousand classified tweets (positive or negative emotionality).

## Working with training set

The training set contains not only the texts of tweets and class tags, but also a large amount of additional information (publication dates, usernames, number of retweets, etc). In the context of this task, this information is not needed, so we leave only texts and labels in the dataset.

Next, process the tweet lines:

* Delete all English words;
* Remove all punctuation (it does not contain any semantic);
* Remove all usernames, tags about retweet (RT) and links;
* Transform all words to lowercase and use the PyMorphy2 module to lemmatize words.

Save the processed dataset to a new cleaned_data file.csv. We will continue to work with it.

## Model building

The main problem is to choose a combination of classifier, vectorization method, n-gram scheme and other parameters in such a way as to maximize the quality of classification.

I chose the two most suitable (for my opinion) models: the naive Bayesian classifier and the linear classifier (minimization by stochastic gradient descent). Sometimes SVM is used for this kind of problems, but it is extremely slow on a large number of objects and features. Logical methods of classification are not considering, because they do not fit this problem at all.

The Bayesian classifier does not need much selection of parameters, but the parameters of the linear model I will select on the grid.

I will also try two methods of vectorizing text data: Count Vectorizer and TF-IDF Vectorizer.

In the end, you will need to choose the optimal scheme of n-grams. Typically, schemes with unigram, bigram or trigram features (and their combinations) are used for such problems.

## The best model

According to the results of validation, the following model was the best:

* N-gram scheme: **(1, 3)** (unigrams + bigrams + trigrams);
* Vectorizer: **TF-IDF**;
* Model type: **linear model**;
* Model parameters: **penalty-l2, alpha-0.000001, loss-log**.

It result was ≈ 0.75.
