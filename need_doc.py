#!/usr/bin/python
# -*- coding: UTF-8 -*-

#import numpy as np
import pandas
import re
import pickle
import pymorphy2

# инициализируем
morph = pymorphy2.MorphAnalyzer()

# функция для лемматизации
def lemmatize(text):
    text_str = str(text)
    text_str = re.sub(r'[^\w\s]+|[\d]+', r'',text_str).strip()
    words = text_str.split() # разбиваем текст на слова
    res = ""
    for word in words:
        p = morph.parse(word)[0]
        res = res + " " + p.normal_form

    return res

if __name__ == "__main__":
    past_columns = []  # пустой массив, для опрееделени колонок
    for i in range(1336):
        past_columns.append(f"v_{i}")

    # ЗАГРУЗКА МОДЕЛИ
    filename = 'finalized_model_resolut.sav'
    loaded_clf = pickle.load(open(filename, 'rb'))

    # ЗАГРУЗКА ВЕКТОРОВ
    filename = 'finalized_vector_resolut.sav'
    loaded_vectorizer = pickle.load(open(filename, 'rb'))

    # РАСЧЕТ С ЗАГРУЖЕННОЙ МОДЕЛИ

    text_predict = 'подготовить ответ'
    text_predict_lemmat = lemmatize(text_predict)
    new_v = loaded_vectorizer.transform([text_predict_lemmat])  # результат
    new_v_array = new_v.toarray()
    dataframe_predict = pandas.DataFrame(data=new_v_array, columns=past_columns)
    need_d = loaded_clf.predict(dataframe_predict)
    print(need_d)

