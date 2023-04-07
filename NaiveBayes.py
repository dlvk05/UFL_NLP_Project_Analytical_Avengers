from sklearn.naive_bayes import MultinomialNB

def naive_bayes(X_train_bow, X_test_bow, y_train, y_test, accuracy_score, confusion_matrix, classification_report):

    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train_bow, y_train)
    y_pred_naiveBayes = naive_bayes_classifier.predict(X_test_bow)

    accuracy = accuracy_score(y_test, y_pred_naiveBayes)
    conf_matrix = confusion_matrix(y_test, y_pred_naiveBayes)
    class_report = classification_report(y_test, y_pred_naiveBayes)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    return naive_bayes_classifier, y_pred_naiveBayes, accuracy, conf_matrix, class_report