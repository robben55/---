import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

data = [
    ("Огромное спасибо, автору за кропотливый труд! Замечательное оформление, детально прорисованные  крупные схемы, очень информативно! Плотная бумага, удобный формат", "Положительный"),
    ("Нужная книга начинающим вязальщикам крючком. Рекомендую.", "Положительный"),
    ("Долгое время занимаюсь вязанием и такие книги отличные помощники. Крючок подзабылся, а книга поможет не только вспомнить, но и вдохновит на новые шедевры!", "Положительный"),
    ("Хорошая книга всё расписано подробно, всё понятно", "Положительный"),
    ("Всё замечательно, информация в книге полезная.", "Положительный"),
    ("Я купила эту книжку как учебное пособие, чтобы связать несложный элемент крючком. Если честно, то представленные схемы не дают нужной информации, вообщем я разочарована", "Отрицательный"),
    ("Дано всего два узора и те без схем, не показано , где можно применить ту или иную схему, не рекомендую к покупке", "Отрицательный"),
    ("Картинки где показано по петлям для меня показались очень мелкими. Пособие неудобно к обучению  ", "Отрицательный"),
    ("Половина страниц оказались залитыми клеем, я разочарована в покупке и буду требовать возврата денег", "Отрицательный"),
    ("Вот я дура, когда моя бабка была жива , могла бы у нее научиться , чем покупать эту бездарную книгу, жалко потраченых 500 рублей", "Отрицательный")
]


X = [x[0] for x in data]
Y = [x[1] for x in data]


def preprocess_text(text):
    stop_words = set(stopwords.words("russian"))
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(word for word in text.split() if word not in stop_words)

X_processed_output = [(preprocess_text(text), tag) for text, tag in zip(X,Y)]

print("\nОбработанные данные:")
for review, tag in X_processed_output:
    print(f"Обзор: {review} | Тэг:{tag}")

X_processed = [preprocess_text(text) for text in X]

X_train, X_test, Y_train, Y_test = train_test_split(X_processed, Y, test_size=0.3, random_state=42)
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
print("\n Уникальные слова:")
print(vectorizer.get_feature_names_out())
print("\nМатрица:")
print(X_train_vectors.toarray())

#
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_bow, Y_train)


Y_pred = model.predict(X_test_bow)
accuracy = accuracy_score(Y_test, Y_pred)


print(f"Точность модели: {accuracy:.2f}")
print("\nОтчет о классификации:")
print(classification_report(Y_test, Y_pred, target_names=["Отрицательный", "Положительный"]))

#

feature_names = vectorizer.get_feature_names_out()
negative_log_probs = model.feature_log_prob_[0]
positive_log_probs = model.feature_log_prob_[1]

negative_word_count = np.exp(negative_log_probs*model.class_count_[0])
positive_word_count = np.exp(positive_log_probs*model.class_count_[1])

total_negative_words = negative_word_count.sum()
total_positive_words = positive_word_count.sum()


freq_table = pd.DataFrame({
    "Word": feature_names,
    "Frequency(Negative)": negative_word_count,
    "Frequency(Positive)": positive_word_count
})

likelihood_table = pd.DataFrame({
    "Word": feature_names,
    "P(Word|Negative)": np.exp(negative_log_probs),
    "P(Word|Positive)": np.exp(positive_log_probs)
})


print("\nFrequency Table:")
print(freq_table)
print("\nLikelihood table:")
print(likelihood_table)



print("===============")


negative_probs = np.exp(negative_log_probs)
positive_probs = np.exp(positive_log_probs)

word_probs = pd.DataFrame({
    "Word": feature_names,
    "P(Word|Negative)": negative_probs,
    "P(Word|Positive)": positive_probs
})

print("\nProbabilities:")
print(word_probs)