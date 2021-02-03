import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import hashing_trick
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords

class Model:
    model = None
    
    def tokenize(self, post):
        """ Words are called tokens and the process of splitting text into tokens is called tokenization. 
        
        This method returns a list of words in a document (post)
        e.g:
            input: "Hello world"
            return: ['hello', 'world']
        """
        return text_to_word_sequence(post, lower=True, split=" ", filters=string.punctuation)
    
    def remove_stop_words(self, post_tokens):
        """remove most common words in a english, eg at, the, a, are, they etc. """
        stop_words = stopwords.words() + ['http', 'com']
        return [word for post in post_tokens for word in post if not word in stop_words]
        
    def train(self, dataset):
        
        # Initialize a learning model
        self.model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
        self.model.compile(optimizer='sgd', loss='mean_squared_error')
        
        #Pre-Process our data
        
        tokenized_posts = [self.tokenize(post) for post in dataset.posts]
        stemmed_words = self.remove_stop_words(tokenized_posts)
        
        
        # The size of the vocabulary defines the hashing space from which words are hashed. 
        # Ideally, this should be larger than the vocabulary by some percentage (perhaps 25%) to minimize the number of collisions.
        
        vocab_size = len(set(stemmed_words))
        rounds = vocab_size*1.3 # total words in our post
        
        encoded_words = [self.encode(word=word, size=rounds) for word in stemmed_words]
        
        # Train the neural net
        self.model.fit(encoded_words, encoded_words, epochs = 500)
    
    def encode(self, size, word):
        """ 
        This function, Integer encodes the content of the post parameter of this function
        
        words [list]
        post [string]
        """
        
        #return one_hot(post, round(size)
        return hashing_trick(word, round(size), hash_function='md5')

    def predict(self):
        return self.model.predict([2030])