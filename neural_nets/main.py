from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
import nets

# загружаем датасет
dataset = pd.read_csv('../data/cleaned_data.csv', index_col=0).dropna()

'''
Необходимо подготовить датасет к работе с сеткой. А именно:

1. Создать словарь всех слов, а в идеале ещё и отранжировать его по
частоте встречаемости слов (в этой работе словарь не ранжировался).

2. Заменить строки на списки целых чисел, которые соответствуют номерам слов
в словаре + 1. Например, предложение "hello world" заменится на [1, 2], если
в словаре слово "hello" имеет индекс 0, а "world" - 1.

3. Применить метод pad_sequences к полученным спискам, чтобы выровнять по длине
численные векторы (наши численные представления строк). Переменная
max_length устанавливает максимальную длину вектора. Недостающие значение будут
обнуляться. Например, после применения метода [[1, 2, 3, 4, 5], [1, 2, 3]]
превратится в [[1, 2, 3, 4, 5], [0, 0, 1, 2, 3]].
'''

# количество объектов в новом датасете
n = 20000
# максимальная длина вектора в новом датасете
max_length = 150

X = []
y = []
dictionary = []

dataset['label'].replace(-1, 0, inplace=True)
dataset = dataset.sample(frac=1)
tpls = list(dataset.itertuples())[:n]

for tpl in tpls:
    row = []
    for word in tpl[1]:
        if word not in dictionary:
            dictionary.append(word)
        row.append(dictionary.index(word) + 1)

    X.append(row)
    y.append(tpl[2])

X = sequence.pad_sequences(np.array(X), maxlen=max_length)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = nets.get_simple_net(len(dictionary), max_length)
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, nb_epoch=1)

# выводим оценку
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
