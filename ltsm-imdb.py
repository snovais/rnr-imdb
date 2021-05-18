# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:42:03 2021

@author: udemy

tarefa de classificação de emoçoes e sentimentos por meio de textos
function: previsão de reviews por meio de Redes Neurais Recorrentes LSTM
"""


import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb

number_of_words = 20000
max_len = 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=number_of_words)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen = max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen = max_len)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim = number_of_words, output_dim = 128, input_shape=(x_train.shape[1],)))

# construcao da LSTM
model.add(tf.keras.layers.LSTM(units = 256, activation = 'tanh'))

# add camada de saída
model.add(tf.keras.layers.Dense(units = 512, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


#compilar o modelo
model.compile(optimizer = 'adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

# treinar o modelo
model.fit(x_train, y_train, epochs = 35, batch_size = 64)

# avaliacao do modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test accuracy: {}".format(test_accuracy))

print("\ntest loss", test_loss)
