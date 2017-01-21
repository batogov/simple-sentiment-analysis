import sklearn
import pandas as pd
import pymorphy2

p_data = pd.read_csv('data/positive.csv', sep=';', header=None)
n_data = pd.read_csv('data/negative.csv', sep=';', header=None)

dataset = pd.concat([p_data, n_data])
dataset = dataset[[3, 4]]

dataset.columns = ['text', 'label']

morph = pymorphy2.MorphAnalyzer()

def text_cleaner(text):
    # к нижнему регистру
    text = text.lower()
    
    # оставляем в предложении только русские буквы (таким образом
    # удалим и ссылки, и имена пользователей, и пунктуацию и т.д.)
    alph = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    
    cleaned_text = ''
    for char in text:
        if (char.isalpha() and char[0] in alph) or (char == ' '):
            cleaned_text += char
        
    result = []
    for word in cleaned_text.split():
        # лемматизируем
        result.append(morph.parse(word)[0].normal_form)
                              
    return ' '.join(result)

dataset['text'] = dataset['text'].apply(text_cleaner)

dataset.to_csv('data/cleaned_data.csv')