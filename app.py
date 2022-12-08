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

app = Flask(__name__, static_folder="static", template_folder="templates")


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')


# Load English data
english_sentences = load_data('data/small_vocab_en.txt')
# Load French data
french_sentences = load_data('data/small_vocab_fr.txt')

df_en = pd.read_csv('data/small_vocab_en.txt', sep = "\t", header = None)
df_fr = pd.read_csv('data/small_vocab_fr.txt', sep = "\t", header = None)

def remove_punc(x):
    return re.sub('[!#?,.:";]', '', x)

df_en[0] = df_en[0].apply(remove_punc)
df_fr[0] = df_fr[0].apply(remove_punc)

def get_maxlen(df):
    maxlen_english = -1
    for doc in df[0].to_list():
        tokens = nltk.word_tokenize(doc)
        if maxlen_english < len(tokens):
            maxlen_english = len(tokens)
    return maxlen_english
# maxlen_french = get_maxlen(df_fr)


# print(maxlen_english)
# print("anglais en haut - francais en bas")
# print(maxlen_french)

def get_y_tokenizer(df):
    y_tokenizer = Tokenizer(char_level=False)
    y_tokenizer.fit_on_texts(df[0])
    return y_tokenizer
y_tokenizer = get_y_tokenizer(df_fr)
# print(y_tokenizer)
# print(y_tokenizer.word_index)
# def preprocessing_prediction(x, maxlen=maxlen_english, df=df_en):
#     maxlen_english = -1
#     for doc in df_en:
#         tokens = nltk.word_tokenize(doc)
#         if maxlen_english < len(tokens):
#             maxlen_english = len(tokens)
#     tokenizer = Tokenizer(char_level = False)
#     tokenizer.fit_on_texts(df)
#     sequences = tokenizer.texts_to_sequences(x)
#     padded = pad_sequences(sequences, maxlen=maxlen, padding = 'post')
#     predictions = model1.predict(padded)[0]
#     id_to_word = {id: word for word, id in y_tokenizer.word_index.items()}
#     id_to_word[0] = ''
#     pred= ' '.join([id_to_word[j] for j in np.argmax(predictions,1)])
#     return pred


def tokenize(x):
    pass


def pad(x, length=None):
    pass

def preprocess(x, y):
   pass
        
# load any model here
model1 = tf.keras.models.load_model('./models/my_model.h5')


def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        if(not request.form['text']):
            return render_template('index.html', error="Please enter a sentence")

        sentence = []
        sentence.append(request.form['text'])

        def preprocessing_prediction(x, df=df_en):
            maxlen_english = get_maxlen(df_en)

            # initialise Tokenizer without the punctuation
            tokenizer = Tokenizer(char_level = False)

            # Fit the tokenizer on the english sentences    
            tokenizer.fit_on_texts(df_en[0].to_list())
            # tokenizer.fit_on_texts(df)

            # Transform the english texts into sequences
            sequences = tokenizer.texts_to_sequences(x)
            
            # Pad the sequences
            padded = pad_sequences(sequences, maxlen=maxlen_english, padding = 'post')
            
            # Predict the french sentences
            predictions = model1.predict(padded)[0]

            # Get the french words
            id_to_word = {id: word for word, id in y_tokenizer.word_index.items()}
            id_to_word[0] = ''
            
            # join the french sentence
            pred= ' '.join([id_to_word[j] for j in np.argmax(predictions,1)])
            return pred
        
        pred = preprocessing_prediction(sentence)
        print(pred)
        return render_template('index.html', original_text=sentence[0], converted_text=pred)


    #     def final_predictions(text):
    #         print("CA PASSE ICI COUZZINNZIZNZNZNZNZNZN")
    #         print(text)
    #         y_id_to_word = {value: key for key, value in french_tokenizer.word_index.items()}
    #         y_id_to_word[0] = '<PAD>'

    #         sentence = [english_tokenizer.word_index[word] for word in text.split()]
            
    #         # sentence = pad_sequences(
    #         #     [sentence], maxlen=preproc_french_sentences.shape[-2], padding='post')
    #         print("ca passe")
    #         print(logits_to_text(simple_rnn_model_wiem.predict(sentence[:1])[0], french_tokenizer))
    #         return (logits_to_text(simple_rnn_model_wiem.predict(sentence[:1])[0], french_tokenizer))

    #     try:
    #         converted_text = final_predictions(request.form['text'].lower())
    #         # if request.form['voice']:
    #         #     return converted_text
    #         print("c le converted text la bravaaa")
    #         print(converted_text)
    #         return render_template('index.html', converted_text=converted_text)
    #     except Exception as e:
    #         print("EXCEPTION", e)
    #         if request.form['voice']:
    #             return "something went wrong"
    #         return render_template('index.html', converted_text="something went wrong")

    return render_template('index.html', converted_text="")


if __name__ == '__main__':
    app.run(debug=True)
