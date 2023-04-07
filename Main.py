import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('/dataset/IMDB_clean.csv')
# data.head()

# function to preprocess the data (tokenize, remove stopwords, and stem words):
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)

# Preprocess the 'review' column in the dataset:
data['cleaned_review'] = data['review'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_review'], data['sentiment_boolean'], test_size=0.2, random_state=42)

# Create a Bag of Words model using CountVectorizer
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

from NaiveBayes import naive_bayes
from RandomForest import random_forest
from SVM import svm
from RNN import rnn
from CNN import cnn

# Dictionary to store the values for the results of different Models
Stats_Dictionary = {}

#Train and Get Results for all the base classifiers
naive_bayes_classifier, y_pred_naiveBayes, accuracy_nb, conf_matrix_nb, class_report_nb = naive_bayes(X_train_bow, X_test_bow, y_train, y_test, accuracy_score, confusion_matrix, classification_report)
Stats_Dictionary['naive_bayes'] = (accuracy_nb, conf_matrix_nb, class_report_nb)

random_forest_classifier, y_pred_rf, accuracy_rf, conf_matrix_rf, class_report_rf  = random_forest(X_train_bow, X_test_bow, y_train, y_test, accuracy_score, confusion_matrix, classification_report)
Stats_Dictionary['random_forest'] = (accuracy_rf, conf_matrix_rf, class_report_rf)

svm_classifier, y_pred_svm, accuracy_svm, conf_matrix_svm, class_report_svm = svm(X_train_bow, X_test_bow, y_train, y_test, accuracy_score, confusion_matrix, classification_report)
Stats_Dictionary['svm'] = (accuracy_svm, conf_matrix_svm, class_report_svm)

accuracy_rnn,model_rnn=rnn(X_train, X_test, y_train, y_test)
accuracy_cnn,model_cnn=cnn(X_train, X_test, y_train, y_test)

Stats_Dictionary["rnn"] = accuracy_rnn

Stats_Dictionary["cnn"] = accuracy_cnn

#Train and Get Results for all the ensemble models
from ensemble1 import ensemble1
from ensemble2 import ensemble2
from ensemble3 import ensemble3

#SVM,NB,RF -Majority Voting
Stats_Dictionary['ensemble1'] = ensemble1(y_test, y_pred_naiveBayes, y_pred_rf, y_pred_svm, accuracy_score, confusion_matrix, classification_report)
#SVM,NB,RF - Stacking
Stats_Dictionary['ensemble2'] = ensemble2(X_train_bow, y_train, y_test, accuracy_score, confusion_matrix, classification_report, y_pred_naiveBayes, y_pred_rf, y_pred_svm, naive_bayes_classifier, random_forest_classifier, svm_classifier)

#CNN ,RNN -Majority Voting
Stats_Dictionary['ensemble3'] = ensemble3(model_rnn, model_cnn,X_train, X_test, y_train, y_test)