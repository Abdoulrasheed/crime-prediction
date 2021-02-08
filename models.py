import numpy as np
import re, string, random
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples as t_sample

# used for training our model
from nltk import classify
from nltk import NaiveBayesClassifier

# reporting library
import plotly.graph_objects as pl

from sklearn.linear_model import LinearRegression

class Model:
    sentiments = []
    
    def __init__(self):
        print("Initializing model...\n")
    
    def convert_to_dict(self, token_list):
        """
            Yield a dictionary with each token as key and it's value True
        """
        
        print("Converting tokenized_words\n")
        
        for tweet_tokens in token_list:
            yield dict([token, True] for token in tweet_tokens)
    
    def count_results(self):
        """ Count the number of positive and negative data from each sentiment """
        
        print("Processing result of nlp\n")
        
        no_of_2015_positive = np.count_nonzero(self.sen_2015 == "Positive")
        print(no_of_2015_positive)
        
        no_of_2015_negative = np.count_nonzero(self.sen_2015 == "Negative")
    
        no_of_2016_positive = np.count_nonzero(self.sen_2016 == "Positive")
        no_of_2016_negative = np.count_nonzero(self.sen_2016 == "Negative")
        
        no_of_2017_positive = np.count_nonzero(self.sen_2017 == "Positive")
        no_of_2017_negative = np.count_nonzero(self.sen_2017 == "Negative")
        
        no_of_2018_positive = np.count_nonzero(self.sen_2018 == "Positive")
        no_of_2018_negative = np.count_nonzero(self.sen_2018 == "Negative")
        
        no_of_2019_positive = np.count_nonzero(self.sen_2019 == "Positive")
        no_of_2019_negative = np.count_nonzero(self.sen_2019 == "Negative")
        
        aggregates = {
                "2015": {"pos": no_of_2015_positive, "neg": no_of_2015_negative}, 
                "2016": {"pos": no_of_2016_positive, "neg": no_of_2016_negative},
                "2017": {"pos": no_of_2017_positive, "neg": no_of_2017_negative},
                "2018": {"pos": no_of_2018_positive, "neg": no_of_2018_negative},
                "2019": {"pos": no_of_2019_positive, "neg": no_of_2019_negative},
                }
        
        return aggregates
            
    def lemmatize_sentence(self, tokens):
        """ 
            analyzes the structure of the word and its context 
            to convert it to a normalized form. eg the word players or playing becomes play.
        """
        
        print("Lemmatizing tokens\n")
        
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
        
        print("Finalizing ...\n")
        
        self.aggs = self.count_results()
        years_pos_sentiments = [[self.aggs['2015']['pos']], [self.aggs['2016']['pos']], [self.aggs['2017']['pos']], [self.aggs['2018']['pos']], [self.aggs['2019']['pos']]]
        years_neg_sentiments = [[self.aggs['2015']['neg']], [self.aggs['2016']['neg']], [self.aggs['2017']['neg']], [self.aggs['2018']['neg']], [self.aggs['2019']['neg']]]
        
        regressor = LinearRegression()
        regressor.fit(years_pos_sentiments, years_neg_sentiments)
        
        self.year_to_predict = input("Please enter the year you wants to predict: ")
        # Predict the rate of crime in the provided year
        return regressor.predict([[float(self.year_to_predict)]])

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
        print("Preparing training data\n")
        
        positive_dataset = [(data_dict, "Positive") for data_dict in positive_tokens_for_model]
        negative_dataset = [(data_dict, "Negative") for data_dict in negative_tokens_for_model]

        # combine our modelled datasets
        dataset = positive_dataset + negative_dataset

        # randomnize the dataset to avoid bias as 
        # the data contains all positive data followed by all negative data in sequence
        random.shuffle(dataset)
        
        return dataset
    
    def plot(self, predicted_result):
        
        years_pos_sentiments = (self.aggs['2015']['pos'], self.aggs['2016']['pos'], self.aggs['2017']['pos'], self.aggs['2018']['pos'], self.aggs['2019']['pos'])
        years_neg_sentiments = (self.aggs['2015']['neg'], self.aggs['2016']['neg'], self.aggs['2017']['neg'], self.aggs['2018']['neg'], self.aggs['2019']['neg'])
        years_pos_sentiments += (abs(predicted_result[0].tolist()[0]/50),)
        years_neg_sentiments += (abs(predicted_result[0].tolist()[0]/165),)
        
        print("Generating graph ...")
        
        predicted_year = f"{self.year_to_predict} Projection"
        
        years = ['2015', '2016', '2017', '2018', '2019', predicted_year]
        x = list(range(len(years)))

        # Specify the plots
        bar_plots = [
            pl.Bar(x=x, y=years_pos_sentiments, name='Positive', marker=pl.bar.Marker(color='#0343df')),
            pl.Bar(x=x, y=years_neg_sentiments, name='Negative', marker=pl.bar.Marker(color='#e50000')),
        ]
        
        # Specify the layout
        layout = pl.Layout(
            title=pl.layout.Title(text="NLP and Machine Learning Crime Prediction", x=0.5),
            yaxis_title="Crime Rate",
            xaxis_tickmode="array",
            xaxis_tickvals=list(range(27)),
            xaxis_ticktext=tuple(years),
        )
            
        # Make the multi-bar plot
        fig = pl.Figure(data=bar_plots, layout=layout)

        # Tell Plotly to render it
        fig.show()
        
        print("Done.")
    
    def tokenize(self):
        print("Tokenizing training data...\n")
        self.pos_data_tokens = t_sample.tokenized('positive_tweets.json')
        self.neg_data_tokens = t_sample.tokenized('negative_tweets.json')
    
    def train(self, dataset):
        pos_data_tokens_list = []
        neg_data_tokens_list = []
        
        self.tokenize()
        
        print("Removing punctuations, stop words and hyperlinks ...\n")
        for p_tokens, n_tokens in zip(self.pos_data_tokens, self.neg_data_tokens):
            pos_data_tokens_list.append(self.remove_noise(p_tokens))
            neg_data_tokens_list.append(self.remove_noise(n_tokens))
        
        positive_tokens_for_model = self.convert_to_dict(pos_data_tokens_list)
        negative_tokens_for_model = self.convert_to_dict(neg_data_tokens_list)
        
        train_data = self.prepare_training_datasets(positive_tokens_for_model, negative_tokens_for_model)
        
        classifier = NaiveBayesClassifier.train(train_data)
        
        for post in dataset.posts:
            tokenised_post = self.remove_noise(word_tokenize(post))
            result = classifier.classify(dict([token, True] for token in tokenised_post))
            self.sentiments.append([post, result])
        
        print("Accuracy is:", classify.accuracy(classifier, train_data))
        print(classifier.show_most_informative_features(15))
            
        
        # split the sentiments into 5 different groups, each representing an imaginary report
        # for a specific year
        splitted = np.array_split(self.sentiments, 5)
        self.sen_2015, self.sen_2016, self.sen_2017, self.sen_2018, self.sen_2019 = splitted