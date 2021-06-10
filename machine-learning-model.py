import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
import tensorflow as tf
import sklearn.model_selection as sk
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenize = Tokenizer()
data = pd.read_csv("/Users/12016/Desktop/Rutgers_Training_Data.csv", engine = "python")

data.head()

X = data["Response"].astype(str)
Y = data["Gibberish"]
tokenize.word_index
tokenize.fit_on_texts(X)

X = tokenize.texts_to_sequences(X)
Z = pad_sequences(X, maxlen=300, padding="post")

Z, Z_test, Y_train, Y_test = sk.train_test_split(Z, Y, test_size=0.25, random_state=1000)

input_dim = Z.shape[1]  

model = keras.Sequential()
model.add(keras.layers.Embedding(151676, 16))
model.add(keras.layers.Dense(16, activation='relu'))
#model.add(keras.layers.GlobalAveragePooling1D())
#model.add(keras.layers.Dropout(rate=0.25))
#model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(Z, Y_train,
    epochs=15,
    verbose=1,
    validation_data=(Z_test, Y_test),
    batch_size=512)

Categories = ["Not Gibberish", "Gibberish"]

result = model.evaluate(Z_test, Y_test)
print(result)

#def prepare(string):
 #   tokenize.fit_on_texts(string)
  #  A = tokenize.texts_to_sequences(string)
   # B = pad_sequences(A, maxlen=300, padding="post")
    #return B
test = 'what is up'
A = tokenize.texts_to_sequences(test)
B = pad_sequences(A
            , maxlen=300, padding="post")

prediction = model.predict(B)
print(prediction[0])