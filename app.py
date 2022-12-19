import os
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import re
import timeit
import pickle

app = Flask(__name__, static_folder="static", template_folder="templates")


# IMPORT dictionnary of list of tokens for each language
with open('./models/tokenizer/english_tokenizer.pickle', 'rb') as handle:
    english_tokenizer = pickle.load(handle)

with open('./models/tokenizer/french_tokenizer.pickle', 'rb') as handle:
    french_tokenizer = pickle.load(handle)
        
# load any model here
model1 = tf.keras.models.load_model('./models/my_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        if(not request.form['text']):
            return render_template('index.html', error="Please enter a sentence")
        print(request.form['text'])
        sentence = []
        sentence.append(request.form['text'])
        
        def preprocessing_prediction(sentence):
            # Transform the english texts into sequences
            sequences = english_tokenizer.texts_to_sequences(sentence)
            # Pad the sequences
            padded = pad_sequences(sequences, maxlen=15, padding = 'post')
            # Predict the french sentences
            predictions = model1.predict(padded)[0]
            # Get the french words
            id_to_word = {id: word for word, id in french_tokenizer.word_index.items()}
            id_to_word[0] = ''
            # join the french sentence
            pred= ' '.join([id_to_word[j] for j in np.argmax(predictions,1)])
            return pred
        
        start = timeit.default_timer()

        pred = preprocessing_prediction(sentence)

        end = timeit.default_timer()

        time_to_predict = end - start
            
        return render_template('index.html', original_text=sentence[0], converted_text=pred, time_to_predict=time_to_predict)

    return render_template('index.html', converted_text="")


if __name__ == '__main__':
    app.run(debug=True)