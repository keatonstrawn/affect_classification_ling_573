"""This script defines a helper class to take the cleaned HatEval data and generate features that a classification
model can use to classify the tweets within the dataset.
"""

# Libraries
import pandas as pd
import scipy.stats as st

from nrclex import NRCLex
from typing import Optional, List
from nltk.tokenize import word_tokenize


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

        # Save normalization info
        self.normalization_dict = {}

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
        rowsums[rowsums == 0] = 1.0
        data.iloc[:, -10:] = data.iloc[:, -10:].div(rowsums, axis=0)

        # ***Uncomment the line below to create file showing the data visualized***
        # data.to_csv('test.txt', sep=',', header=True)

        return data

    def normalize_feature(self, data: pd.DataFrame, feature_columns: List[str], normalization_method: Optional[str] = None):
        """Normalizes the features in the specified columns by transforming the data to fall within [0,1].

        This can be done using a number of different approaches. The specific approach and relevant parameters needed to
        perform normalization in downstream transformations are saved in the normalization_dict, which is keyed by the
        feature column name.

        Arguments:
        ----------
        data
            The data for which the feature is to be generated
        feature_columns
            The column name(s) of the feature(s) to be normalized. If multiple column names are provided then the values
            in both columns are simultaneously normalized.
        normalization_method
            Required when fitting. Specifies which calculation to use to normalize the features. Options include...
            min_max:
                Applies (x-min)/(max-min) transformation, capping the resulting values to fall within [0,1].
            z_score:
                Applies Norm.CDF((x-mu)/sigma) transformation. Values correspond to percentages from a normal
                distribution.
                #TODO: extend this method to use a more appropriate distribution than normal for certain features

        Returns:
        -------
        transformed_data
            The original train_data dataframe with new columns that include the normalized features for each observation
            in the dataset.
        """

        # Initialize dictionary to hold normalized results
        normalized_feats = {}

        # Perform normalization transformations, assuming fitting has already occurred
        if self.fitted:
            for feat in feature_columns:

                # If trained normalization method uses min-max approach
                if self.normalization_dict[feat]['method'] == 'min_max':
                    f_min = self.normalization_dict[feat]['params']['min']
                    f_max = self.normalization_dict[feat]['params']['max']
                    feat_vals = data[feat]
                    norm_vals = (feat_vals - f_min) / (f_max - f_min)

                # If trained normalization method uses z-score approach
                if self.normalization_dict[feat]['method'] == 'z_score':
                    sigma = self.normalization_dict[feat]['params']['sigma']
                    mu = self.normalization_dict[feat]['params']['mu']
                    feat_vals = data[feat]
                    z_scores = (feat_vals - mu) / sigma
                    norm_vals = st.norm.cdf(z_scores)

                # Store results to be returned
                normalized_feats[feat] = norm_vals

        # Learn and apply normalization transformations
        else:
            for feat in feature_columns:
                self.normalization_dict[feat] = {}
                feat_vals = data[feat]

                # If specified normalization method is min-max approach
                if normalization_method == 'min_max':
                    f_min = feat_vals.min()
                    f_max = feat_vals.max()
                    norm_vals = (feat_vals - f_min) / (f_max - f_min)
                    # Save parameters for future transformations
                    self.normalization_dict[feat]['method'] = 'min_max'
                    self.normalization_dict[feat]['params'] = {'min': f_min, 'max': f_max}

                # If specified normalization method is z-score approach
                if normalization_method == 'z_score':
                    sigma = feat_vals.std()
                    mu = feat_vals.mean()
                    z_scores = (feat_vals - mu) / sigma
                    norm_vals = st.norm.cdf(z_scores)
                    # Save parameters for future transformations
                    self.normalization_dict['method'] = 'z_score'
                    self.normalization_dict['params'] = {'sigma': sigma, 'mu': mu}

                # Store results to be returned
                normalized_feats[feat] = norm_vals

        # Add normalized features to dataframe
        n_cols = len(data.columns)
        for k in normalized_feats.keys():
            2+2

        # n_cols = len(pred_df.columns)
        # for t in task_cols:
        #     pred_df.insert(loc=n_cols, column=f'{t}_prediction', value=y_pred[t].values)
        #     n_cols += 1

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


