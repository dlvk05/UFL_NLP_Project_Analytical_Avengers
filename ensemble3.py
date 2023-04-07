import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Combine the predictions using averaging
    # majority voting approach is more applicable to a multi-class classification problem. 
    # Since this is a binary classification, 
    # we will use a simple averaging approach instead.
def combine_predictions(predictions_list):
        combined_proba = np.mean(predictions_list, axis=0)
        combined_pred = (combined_proba > 0.5).astype("int32")
        return combined_pred

def ensemble3(model_rnn,model_cnn,X_train, X_test, y_train, y_test):
    # Set up parameters for tokenization and padding:
    vocab_size = 10000
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'
    oov_token = '<OOV>'
    max_words = 10000

    # Tokenize and pad the sequences:
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

    y_pred_rnn = (model_rnn.predict(X_test_pad) > 0.5).astype("int32")
    y_pred_cnn = (model_cnn.predict(X_test_pad) > 0.5).astype("int32")

    #Combine the predictions using averaging
    y_pred_combined = combine_predictions([y_pred_rnn, y_pred_cnn])

    # Evaluate the ensemble model
    ensemble_accuracy = accuracy_score(y_test, y_pred_combined)
    print(f'Ensemble accuracy: {ensemble_accuracy}')
    return ensemble_accuracy