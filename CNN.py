#using CNN
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense



def cnn(X_train, X_test, y_train, y_test):
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

    # Define and compile the RNN model:
    embedding_dim = 100

    model_cnn = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_length),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model:
    model_cnn.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    # Train the CNN  model using the training data:
    history = model_cnn.fit(X_train_pad, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1)
    
    # Evaluate the model
    loss, accuracy = model_cnn.evaluate(X_test_pad, y_test)
    print(f'Test accuracy: {accuracy}')

    return (accuracy)



def rnn(X_train, X_test, y_train, y_test, accuracy_score, confusion_matrix, classification_report):
    # Set up parameters for tokenization and padding:
    vocab_size = 10000
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'
    oov_token = '<OOV>'

    # Tokenize and pad the sequences:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(X_train)

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Define and compile the RNN model:
    embedding_dim = 16

    model_cnn = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        SimpleRNN(32),
        Dense(1, activation='sigmoid')
    ])

    model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the RNN model using the training data:
    epochs = 30
    history = model_cnn.fit(X_train_padded, y_train, epochs=epochs, validation_data=(X_test_padded, y_test), verbose=2)

    # Evaluate the model's performance:
    loss, accuracy_rnn = model_cnn.evaluate(X_test_padded, y_test)
    print(f"RNN Accuracy: {accuracy_rnn}")

    return (accuracy_rnn,model_cnn)