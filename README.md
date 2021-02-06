#Crime Prediction

### TO DOWNLOAD Stop Words

```py

import nltk

nltk.download('stopwords')

```

Model Architecture:

Embedding Layer: in simple terms, it creates word vectors of each word in the word_index and group words that are related or have similar meaning by analyzing other words around them.

LSTM Layer: to make a decision to keep or throw away data by considering the current input, previous output, and previous memory. There are some important components in LSTM.

Forget Gate, decides information is to be kept or thrown away
Input Gate, updates cell state by passing previous output and current input into sigmoid activation function
Cell State, calculate new cell state, it is multiplied by forget vector (drop value if multiplied by a near 0), add it with the output from input gate to update the cell state value.
Ouput Gate, decides the next hidden state and used for predictions
Dense Layer: compute the input with the weight matrix and bias (optional), and using an activation function. I use Sigmoid activation function for this work because the output is only 0 or 1.

The optimizer is Adam and the loss function is Binary Crossentropy because again the output is only 0 and 1, which is a binary number.
