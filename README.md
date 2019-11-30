# Sentiment Analysis

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

## Working with set

Выборка содержит в себе не только тексты твитов и метки классов, но и большое количество дополнительной информации (даты публикаций, имена пользователей, количество ретвитов и т.д.). В контексте данной задачи эта информация не нужна, поэтому оставляем в датасете только тексты и метки.

Далее обработаем строки твитов. А именно:

* Удалим английские слова;
* Удалим всю пунктуацию (в коротких текстах твитов она не несёт какую-либо смысловую нагрузку);
* Удалим из твитов имена пользователей, метки о ретвитте (RT) и ссылки;
* Приведём все слова к нижнему регистру и с помощью модуля PyMorphy2 произведём лемматизацию слов.

Сохраним обработанный датасет в новый файл _cleaned_data.csv_. Дальше работать будем уже с ним.

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
