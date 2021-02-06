import re, string, random
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples as t_sample

# used for training our model
from nltk import classify
from nltk import NaiveBayesClassifier

class Model:
    model = None
    
    def convert_to_dict(self, token_list):
        """
            Yield a dictionary with each token as key and it's value True
        """
        
        for tweet_tokens in token_list:
            yield dict([token, True] for token in tweet_tokens)
            
    def lemmatize_sentence(self, tokens):
        """ 
            analyzes the structure of the word and its context 
            to convert it to a normalized form. eg the word players or playing becomes play.
        """
        
        lemmatizer = WordNetLemmatizer()
        lemmatized_sentence = []
        for word, tag in pos_tag(tokens):
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
        return lemmatized_sentence

    def predict(self):
        return self.model.predict([2030])

    def remove_noise(self, tweet_tokens):
        
        """
            Remove punctuations, stop words and hyperlinks 
        """

        cleaned_tokens = []
        stop_words = stopwords.words('english')

        for token, tag in pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens
    
    def prepare_training_datasets(self, positive_tokens_for_model, negative_tokens_for_model):
        positive_dataset = [(data_dict, "Positive") for data_dict in positive_tokens_for_model]
        negative_dataset = [(data_dict, "Negative") for data_dict in negative_tokens_for_model]

        # combine our modelled datasets
        dataset = positive_dataset + negative_dataset

        # randomnize the dataset to avoid bias as 
        # the data contains all positive data followed by all negative data in sequence
        random.shuffle(dataset)
        
        return dataset
    
    def tokenize(self):
         self.pos_data_tokens = t_sample.tokenized('positive_tweets.json')
         self.neg_data_tokens = t_sample.tokenized('negative_tweets.json')
    
    def train(self, dataset):
        pos_data_tokens_list = []
        neg_data_tokens_list = []
        
        
        self.tokenize()
        
        for p_tokens, n_tokens in zip(self.pos_data_tokens, self.neg_data_tokens):
            pos_data_tokens_list.append(self.remove_noise(p_tokens))
            neg_data_tokens_list.append(self.remove_noise(n_tokens))
        
        positive_tokens_for_model = self.convert_to_dict(pos_data_tokens_list)
        negative_tokens_for_model = self.convert_to_dict(neg_data_tokens_list)
        
        train_data = self.prepare_training_datasets(positive_tokens_for_model, negative_tokens_for_model)
        
        classifier = NaiveBayesClassifier.train(train_data)
        
        sentiments = []
        
        for post in dataset.posts:
            tokenised_post = self.remove_noise(word_tokenize(post))
            result = classifier.classify(dict([token, True] for token in tokenised_post))
            sentiments.append({"post": post, "result": result})
        
        print(sentiments)
            