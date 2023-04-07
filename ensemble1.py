#Ensemble SVM,Naive Bayies , Random Forest-Majority Voting
def ensemble1(y_test, y_pred_naiveBayes, y_pred_rf, y_pred_svm, accuracy_score, confusion_matrix, classification_report):
    #Ensemble SVM,Naive Bayies , Random Forest -Majority Voting
    # Combine the predictions of the three classifiers for each instance in the test data:
    predictions = list(zip(y_pred_naiveBayes, y_pred_rf, y_pred_svm))

    # function that returns the majority sentiment
    def majority_voting(predictions):
        return round(sum(predictions) / len(predictions))
    
    y_pred_ensemble = [majority_voting(pred_tuple) for pred_tuple in predictions]

    # Evaluate the ensemble model's performance:
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
    conf_matrix_ensemble = confusion_matrix(y_test, y_pred_ensemble)
    class_report_ensemble = classification_report(y_test, y_pred_ensemble)

    print(f"Ensemble Accuracy: {accuracy_ensemble}")
    print(f"Ensemble Confusion Matrix:\n{conf_matrix_ensemble}")
    print(f"Ensemble Classification Report:\n{class_report_ensemble}")

    return (accuracy_ensemble, conf_matrix_ensemble, class_report_ensemble)