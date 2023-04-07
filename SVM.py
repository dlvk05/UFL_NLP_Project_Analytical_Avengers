from sklearn.svm import LinearSVC

def svm(X_train_bow, X_test_bow, y_train, y_test, accuracy_score, confusion_matrix, classification_report):
    # train the SVM model using the training data 
    svm_classifier = LinearSVC(random_state=42)
    svm_classifier.fit(X_train_bow, y_train)

    y_pred_svm = svm_classifier.predict(X_test_bow)

    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    class_report_svm = classification_report(y_test, y_pred_svm)

    print(f"SVM Accuracy: {accuracy_svm}")
    print(f"SVM Confusion Matrix:\n{conf_matrix_svm}")
    print(f"SVM Classification Report:\n{class_report_svm}")

    return svm_classifier, y_pred_svm, accuracy_svm, conf_matrix_svm, class_report_svm