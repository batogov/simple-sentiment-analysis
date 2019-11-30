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

## Построение моделей

Задача заключается в том, чтобы выбрать сочетание классификатора, метода векторизации, схемы n-грамм и прочие параметры таким образом, чтобы максимизировать качество классификации.

Я выбрал две наиболее подходящие (по моему мнению) модели: наивный байесовский классификатор и линейный классификатор (минимизация с помощью стохастического градиентного спуска). Иногда для подобного рода задач используют SVM, но он крайне медленно работает на большом количестве объектов и признаков. Логические методы классификации и вовсе не рассматриваются, так как совершенно не подходят к данной задаче.

Байесовский классификатор особо не нуждается в подборе параметров, а вот параметры линейной модели я буду подбирать по сетке.

Так же я буду пробовать два метода векторизации текстовых данных: Count Vectorizer и TF-IDF Vectorizer. 

Последнее, что нужно будет оптимально подобрать – схему n-грамм. Обычно для таких задач используют схемы с униграммными, биграмными или триграммными признаками, а также их совместные комбинации.

## Лучшая модель

По результатам валидации, лучшей оказалась следующая модель:

* Схема n-грамм: **(1, 3)** (униграммы + биграммы + триграммы);
* Vectorizer: **TF-IDF**;
* Тип модели: **линейная модель**;
* Параметры модели: **penalty – l2, alpha – 0.000001, loss – log**.

Её результат оказался ≈ 0.75.
