#Ensemble SVM,Naive Bayies , Random Forest-Stacking
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import numpy as np

def ensemble2(X_train_bow, y_train, y_test, accuracy_score, confusion_matrix, classification_report, y_pred_naiveBayes, y_pred_rf, y_pred_svm, naive_bayes_classifier, random_forest_classifier, svm_classifier):

    # Prepare the data for stacking by using the predictions of the three classifiers on the training dataset:
    y_train_pred_naiveBayes = cross_val_predict(naive_bayes_classifier, X_train_bow, y_train, cv=5)
    y_train_pred_rf = cross_val_predict(random_forest_classifier, X_train_bow, y_train, cv=5)
    y_train_pred_svm = cross_val_predict(svm_classifier, X_train_bow, y_train, cv=5)

    train_predictions = np.column_stack((y_train_pred_naiveBayes, y_train_pred_rf, y_train_pred_svm))

    # Train the meta-model (a logistic regression classifier) on the stacked training predictions:
    meta_model = LogisticRegression()
    meta_model.fit(train_predictions, y_train)

    # Prepare the data for stacking on the test dataset by using the predictions of the three classifiers:
    test_predictions = np.column_stack((y_pred_naiveBayes, y_pred_rf, y_pred_svm))

    # Calculate the ensemble predictions by applying the meta-model to the stacked test predictions:
    y_pred_stacking = meta_model.predict(test_predictions)

    # Evaluate the stacking ensemble model's performance:
    accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
    conf_matrix_stacking = confusion_matrix(y_test, y_pred_stacking)
    class_report_stacking = classification_report(y_test, y_pred_stacking)

    print(f"Stacking Accuracy: {accuracy_stacking}")
    print(f"Stacking Confusion Matrix:\n{conf_matrix_stacking}")
    print(f"Stacking Classification Report:\n{class_report_stacking}")

    return (accuracy_stacking, conf_matrix_stacking, class_report_stacking)