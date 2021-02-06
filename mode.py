from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb

def get_data():
    names = ['fullname', 'phone', 'crime', 'address', 'posts']
    try:
        return pd.read_csv('datasets/crime_main.csv', names=names, keep_default_na=False, na_values=[""])
    except:
        return None

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results.astype('float32')

cdata = get_data()
train_data = cdata.posts

# Our vectorized training data
x_train = vectorize_sequences(train_data)

# Our vectorized test data
x_test = vectorize_sequences(test_data)

# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model = keras.models.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000,), name='reviews'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
estimator_model = keras.estimator.model_to_estimator(keras_model=model)

def input_function(features,labels=None,shuffle=False,epochs=None,batch_size=None):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"reviews_input": features},
        y=labels,
        shuffle=shuffle,
        num_epochs=epochs,
        batch_size=batch_size
    )
    return input_fn

estimator_model.train(input_fn=input_function(partial_x_train, partial_y_train, True, 20,512))
score = estimator_model.evaluate(input_function(x_val, labels=y_val))
print(score)