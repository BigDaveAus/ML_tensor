import os
#!pip install numpy tensorflow scikit-learn
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Embedding, Dense, Flatten
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import 
from datetime import datetime

# Set up logging
logging.basicConfig(filename='script_output.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

learning_counter = 0

# Prepare data set
with open('learning_data.pkl', 'rb') as file:
    learning_data = pickle.load(file)
combined_text, combined_label = learning_data
print("Data loaded successfully. Number of samples: ", len(combined_label), ".\n\
      Preparing to separate samples...")
logging.info("Data loaded successfully. Number of samples: %d. Preparing to separate samples...", len(combined_label))


def separate_samples(combined_text, combined_label):
    X_train, X_sample, y_train, y_sample = train_test_split(
        combined_text,   # Features
        combined_label,  # Labels
        test_size=0.1,   # 10% of the data goes into the test set
        stratify=combined_label,  # Stratify by labels to maintain label proportions
        random_state=42  # Seed for reproducibility
    )
    print("Samples separated successfully.")
    logging.info("Samples separated successfully.")
    return X_train, X_sample, y_train, y_sample


def tokenize_text(X_train, X_sample):
    maxlen = 250  # Cuts off reviews after 150 words
    max_words = 10000000  # Considers only the top 100,000 words in the dataset
    #have expanded the num words for improved accuracy
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_sample)

    x_train = pad_sequences(sequences_train, maxlen=maxlen)
    x_test = pad_sequences(sequences_test, maxlen=maxlen)
    print("Text tokenized successfully.")
    logging.info("Text tokenized successfully.")
    
    return x_train, x_test, maxlen, max_words


def initial_model(x_train, y_train, x_test, y_test, maxlen, max_words):
    dt = datetime.now().strftime('%Y%m%D%H%M%S')
    model = Sequential()
    model.add(Embedding(max_words, 8))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    results = model.evaluate(x_test, y_test)
    model.save(f'initial_model{dt}.keras')
    print(f'Test accuracy: {results[1]}, MSE: {results[2]}.')
    logging.info(f'Test accuracy: {results[1]}, MSE: {results[2]}.')
    return model, history, results

"""
def further_training(model, results, x_train, y_train, x_test, y_test):
    global learning_counter
    learning_counter += 1
    new_model = clone_model(model)
    new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])
    history = new_model.fit(x_train, y_train, epochs=(learning_counter*2), batch_size=32, validation_split=0.1)
    new_results = new_model.evaluate(x_test, y_test)
    print(f'Further training {learning_counter} completed. Test accuracy: {new_results[1]}, MSE: {new_results[2]}.')
    logging.info(f'Further training {learning_counter} completed. Test accuracy: {new_results[1]}, MSE: {new_results[2]}.')
    if new_results[1] < results[1]:
        model = new_model
        results = new_results
        model.save(f'further_training_{learning_counter}.keras')
        print(f'Further training {learning_counter} model saved.')
        logging.info(f'Further training {learning_counter} model saved.')
        further_training(model, results, x_train, y_train, x_test, y_test)
    else:
        new_model.save(f'Final_training_{learning_counter}.keras')
        print(f'Further training stopped. Best model saved as Final_training_{learning_counter}.keras')
        logging.info(f'Further training stopped. Best model saved as Final_training_{learning_counter}.keras')
        
        return results, model, history
"""

def further_training(model, results, x_train, y_train, x_test, y_test):
    dt = datetime.now().strftime('%Y%m%D%H%M%S')
    global learning_counter
    learning_counter += 1
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'mse'])
    history = model.fit(x_train, y_train, epochs=(learning_counter*2), batch_size=64, validation_split=0.1)
    results = model.evaluate(x_test, y_test)
    model.save(f'further_training_{learning_counter}_{dt}.keras')
    print(f'Further training {learning_counter}_{dt} completed. Test accuracy: {new_results[1]}, MSE: {new_results[2]}.')
    logging.info(f'Further training {learning_counter}_{dt} completed. Test accuracy: {new_results[1]}, MSE: {new_results[2]}.')
    return results, model, history
        #consider using while loop instead of recursion

def train_further(model, results, x_train, y_train, x_test, y_test):
    while learning_counter < 10:
        while results[1] < 0.9:
            results, model, history = further_training(model, results, x_train, y_train, x_test, y_test)
        else:
            model.save(f'final_model_{dt}.keras')
            print(f'Further training stopped. Best model saved as final_model_{dt}.keras')
            logging.info(f'Further training stopped. Best model saved as final_model_{dt}.keras')
            break



def main():
    try:
        logging.info('Script started')
        X_train, X_sample, y_train, y_sample = separate_samples(combined_text, combined_label)
        x_train, x_test, maxlen, max_words = tokenize_text(X_train, X_sample)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_sample)
        model, history, results = initial_model(x_train, y_train, x_test, y_test, maxlen, max_words)
        train_further(model, results, x_train, y_train, x_test, y_test)
        logging.info('Script finished successfully')

    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise


def send_email():
    from_addr = 'notifications@datasciencematters.com.au'
    to_addr = 'david.hayblum@gmail.com'
    subject = 'Script Finished'
    
    with open('script_output.log', 'r') as file:
        log_content = file.read()

    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = subject

    msg.attach(MIMEText(log_content, 'plain'))

    try:
        server = smtplib.SMTP('mail.datasciencematters.com.au', 587)  # Replace with your SMTP server details
        server.starttls()
        server.login(from_addr, 'bailey')  # Replace with your email account details
        text = msg.as_string()
        server.sendmail(from_addr, to_addr, text)
        server.quit()
        logging.info('Email sent successfully')
    except Exception as e:
        logging.error(f'Failed to send email: {e}')


if __name__ == '__main__':
    main()
    send_email()
