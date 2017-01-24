from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit, cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import pandas as pd


# считываем подготовленный датасет
dataset = pd.read_csv('data/cleaned_data.csv', index_col=0).dropna()

# массив n-граммных схем, которые будут использоваться в работе
# например, (1, 3) означает униграммы + биграммы + триграммы
ngram_schemes = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]

for ngram_scheme in ngram_schemes:

	print('N-gram Scheme:', ngram_scheme)

	count_vectorizer = CountVectorizer(analyzer = "word", ngram_range=ngram_scheme) 
	tfidf_vectorizer = TfidfVectorizer(analyzer = "word", ngram_range=ngram_scheme)

	vectorizers = [count_vectorizer, tfidf_vectorizer]
	vectorizers_names = ['Count Vectorizer', 'TF-IDF Vectorizer']

	for i in range(len(vectorizers)):
		print(vectorizers_names[i])
		vectorizer = vectorizers[i]

		X = vectorizer.fit_transform(dataset['text'])
		y = dataset['label']

		cv = ShuffleSplit(len(y), n_iter=5, test_size=0.3, random_state=0)

		# наивный байес
		clf = MultinomialNB()
		NB_result = cross_val_score(clf, X, y, cv=cv).mean()

		# линейный классификатор
		clf = SGDClassifier()

		parameters = {
		    'loss': ('log', 'hinge'),
		    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
		    'alpha': [0.001, 0.0001, 0.00001, 0.000001]
		}

		gs_clf = GridSearchCV(clf, parameters, cv=cv, n_jobs=-1)
		gs_clf = gs_clf.fit(X, y)

		L_result = gs_clf.best_score_

		print('NB:', NB_result.mean())
		print('Linear:', L_result)
		print('Linear Parameters:', gs_clf.best_params_)
		print()