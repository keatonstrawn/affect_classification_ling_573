"""This script defines a helper class to take the cleaned HatEval data and generate features that a classification
model can use to classify the tweets within the dataset.
"""

# Libraries
import torch
import numpy as np
import pandas as pd

from nrclex import NRCLex
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase, PreTrainedModel
from typing import List, Union, Optional, Dict

# for Universal Sentence Encoder -- need to add tensorflow to environment.yml file                   
#import tensorflow as tf
#import tensorflow_hub as hub


# Define class to perform feature engineering
class FeatureEngineering:

    def __init__(self):
        """Generates features from processed data to be used in hate speech detection tasks A and B, as specified in
        SemEval 2019 task 5.

        Includes methods to generate the following features:
            * _NRC_counts --> binary classification of words across ten emotional dimensions. Raw counts are then
                                    transformed into proportions to normalize across tweets.
            * example feature2 --> fill this in with actual feature
            * example feature3 --> fill this in with actual feature
        """

        # Initialize the cleaned datasets
        self.train_data: Optional[pd.DataFrame] = None

        # Set fit flag
        self.fitted = False

    def _NRC_counts(self, data: pd.DataFrame):
        """This method uses data from the NRC Word-Emotion Association Lexicon, which labels words with either a 1 or 0 based on
        the presence or absence of each of the following emotional dimensions: anger, anticipation, disgust, fear, joy, negative, 
        positive, sadness, surprise, trust. It sums the frequency counts in each of the ten dimensions across all the words 
        in a tweet, then divides by the total number of counts to obtain a proportion. These proportions are added on to the end 
        of the dataframe as count-based features.


        Arguments:
        ---------
        data
            The data for which the feature is to be generated

        Returns:
        -------
        The original dataset with ten new columns that contain the new emotion features generated for each tweet in the
        dataset.

        """
        # add ten columns to the end of the dataframe, representing the eight emotional dimensions of NRC
        emotions = ['negative', 'positive', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        for emotion in emotions:
            data[emotion] = 0
        
        # iterate over each tweet to get counts of each emotion classification
        for index, row in data.iterrows():
            text = word_tokenize(row['cleaned_text'])

            # iterate over each word in the tweet, add counts to emotion vector
            for word in text:
                emotion = NRCLex(word)
                if emotion is not None:
                    emolist = emotion.affect_list
                    for emo in emolist:
                        data.at[index, emo] += 1

        # divide by total count of emo markers to get proportions not frequency counts
        # Replace 0 values with NaN to prevent error with dividing by zero
        rowsums = data.iloc[:, -10:].sum(axis=1)
        rowsums[rowsums == 0] = float('NaN')
        data.iloc[:, -10:] = data.iloc[:, -10:].div(rowsums, axis=0)

        # ***Uncomment the line below to create file showing the data visualized***
        # data.to_csv('test.txt', sep=',', header=True)

        return data
    
    def embeddings_helper(self, tweet: str, model: Union[Dict, KeyedVectors, PreTrainedModel], embedding_type: str, 
                          tokenizer: Optional[PreTrainedTokenizerBase] = None) -> List[List[float]]:
        """Helper function to get FastText, BERTweet, or GloVe embeddings. Tokenizes input and accesses embeddings
        from model/dictionary.

        Arguments:
        ---------
        tweet
            The line of the data to generate embeddings for
        model
            The dictionary of FastText embeddings, as either a Dict or an instance of KeyedVectors from gensim
        embedding_type
            '1' == FastText
            '2' == BERTweet
            '3' (or anything else) == GloVe
        tokenizer
            Optional Tokenizer for BERTweet embeddings

        Returns:
        -------
        A list of the word embeddings for each word in the input tweet.

        """
        # tokenize
        words = tweet.split()
        
        # retrieve embeddings if in the vocabulary/model
        if embedding_type == '1':
            embeddings = [model[word] for word in words if word in model.key_to_index]
        elif embedding_type == '2':
            # different form of tokenizing
            tokens = tokenizer.tokenize(tweet)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            with torch.no_grad():
                outputs = model(torch.tensor([input_ids]))
            embed = outputs.last_hidden_state[0]
            embed_np = embed.detach().numpy()
            embeddings = [embed_np[i].tolist() for i in range(len(tokens))]
        else:
            embeddings = [model[word] for word in words if word in model.keys()]
            
        return embeddings

    def get_fasttext_embeddings(self, df: pd.DataFrame, embedding_file_path: str):
        """Function to get FastText embeddings from a dataframe and automatically add them to this dataframe. These
        are pretrained embeddings with d_e == 300 

        Arguments:
        ---------
        df
            Pandas dataframe containing the preprocessed data
        embedding_file_path
            File path for the embeddings file

        Returns:
        -------
        Nothing

        """
        # get the model from a preloaded corpus
        model = KeyedVectors.load_word2vec_format(embedding_file_path)

        # get the embeddings for each row and save to a new column in the dataframe
        df['fastText_embeddings'] = df['cleaned_text'].apply(lambda tweet: self.embeddings_helper(tweet, model, '1'))

    def get_bertweet_embeddings(self, df: pd.DataFrame):
        """Function to get BERTweet embeddings from a dataframe and automatically add them to this dataframe.
        These embeddings are learned from a model, with d_e == ?? (probably should know this)

        Arguments:
        ---------
        df
            Pandas dataframe containing the preprocessed data

        Returns:
        -------
        Nothing

        """
        # load tokenizer and model
        model = AutoModel.from_pretrained("vinai/bertweet-base")
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

        # get the embeddings for each row and save to a new column in the dataframe
        df['BERTweet_embeddings'] = df['cleaned_text'].apply(lambda tweet: self.embeddings_helper(tweet, model, '2', tokenizer))

    def get_glove_embeddings(self, df: pd.DataFrame, embedding_file_path: str):
        """Function to get GloVe embeddings from a dataframe and automatically add them to this dataframe. These
        are pretrained embeddings with d_e == 300.

        Arguments:
        ---------
        df
            Pandas dataframe containing the preprocessed data
        embedding_file_path
            File path for the embeddings file

        Returns:
        -------
        Nothing

        """
        # load embeddings and make a dict
        embeddings_index = {}
        with open(embedding_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        # get the embeddings for each row and save to a new column in the dataframe
        df['GloVe_embeddings'] = df['cleaned_text'].apply(lambda tweet: self.embeddings_helper(tweet, embeddings_index, '3'))    

    def get_universal_sent_embeddings(self, df: pd.DataFrame):
        """Function to get Google Universal Sentence Encoder embeddings from a dataframe and automatically add them
        to this dataframe. These embeddings are for a whole sentence rather than for individual words and are of 
        d_e == 512. 

        Arguments:
        ---------
        df
            Pandas dataframe containing the preprocessed data

        Returns:
        -------
        Nothing

        """
        # load the embeddings from tensorflow hub
        embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

        # get the embeddings for each row and save to a new column in the dataframe
        df['Universal_Sentence_Encoder_embeddings'] = df['cleaned_text'].apply(embed)

    def fit_transform(self, train_data):
        """Learns all necessary information from the provided training data in order to generate the complete set of
        features to be fed into the classification model. In the fitting process, the training data is also transformed
        into the feature-set expected by the model and returned.

        Arguments:
        ---------
        train_data
            The training data that is used to define the feature-engineering methods.

        Returns:
        -------
        transformed_data
            The original train_data dataframe with new columns that include the calculated features for each observation
            in the dataset.
        """

        # Get the training data, to be used for fitting
        self.train_data = train_data

        # Framework to add in steps for each feature that is to be generated
        # transformed_data = self._example_feature1_method(train_data, fit=True, other_args=None)
        transformed_data = self._NRC_counts(train_data)

        # TODO: add in code below to fit and transform training data to generate other features as they are added

        # Update the fitted flag
        self.fitted = True

        return transformed_data

    def transform(self, data):
        """Uses the feature-generating methods that were fit in an earlier step to transform a new dataset to include
        the feature-set expected by the classification model.

        Arguments:
        ---------
        data
            The data set for which the feature set is to be generated.

        Returns:
        -------
        transformed_data
            The original dataframe with new columns that include the calculated features for each observation in the
            dataset.
        """

        # Ensure feature generating methods have been trained prior to transforming the data
        assert self.fitted, 'Must apply fit_transform to training data before other datasets can be transformed.'

        # Framework to add in steps for each feature that is to be generated
        transformed_data = self._NRC_counts(data)

        # TODO: add in code below to transform datasets to generate other features as they are added

        return transformed_data


if __name__ == '__main__':

    # Imports
    from data_processor import DataProcessor

    # Load and clean the raw data
    myDP = DataProcessor()
    myDP.load_data(language='english', filepath='../data')  # May need to change to './data' or 'data' if on a Mac
    myDP.clean_data()

    # Instantiate the FeatureEngineering object
    myFE = FeatureEngineering()

    # Fit
    train_df = myFE.fit_transform(myDP.processed_data['train'])

    # Transform
    val_df = myFE.transform(myDP.processed_data['validation'])

    # View a sample of the results
    train_df.head()
    val_df.head()


