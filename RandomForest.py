# Random Forest
from sklearn.ensemble import RandomForestClassifier

def random_forest(X_train_bow, X_test_bow, y_train, y_test, accuracy_score, confusion_matrix, classification_report):

    random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_classifier.fit(X_train_bow, y_train)

    # Predict the sentiment of the test data and evaluate the model 
    y_pred_rf = random_forest_classifier.predict(X_test_bow)

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    class_report_rf = classification_report(y_test, y_pred_rf)

    print(f"Random Forest Accuracy: {accuracy_rf}")
    print(f"Random Forest Confusion Matrix:\n{conf_matrix_rf}")
    print(f"Random Forest Classification Report:\n{class_report_rf}")

    return random_forest_classifier, y_pred_rf, accuracy_rf, conf_matrix_rf, class_report_rf