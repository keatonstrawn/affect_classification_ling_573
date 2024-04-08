""" This script defines a helper class to take the cleaned HatEval data and generate features that a classification
model can use to classify the tweets within the dataset.
"""

# Libraries
import pandas as pd

from typing import Optional


# Define class to perform feature engineering
class FeatureEngineering:

    def __init__(self):
        """Generates features from processed data to be used in hate speech detection tasks A and B, as specified in
        SemEval 2019 task 5.

        Includes methods to generate the following features:
            * example feature1 --> fill this in with actual feature
            * example feature2 --> fill this in with actual feature
            * example feature3 --> fill this in with actual feature
        """

        # Initialize the cleaned datasets
        self.train_data: Optional[pd.DataFrame] = None

        # Set fit flag
        self.fitted = False

    def _example_feature1_method(self, data: pd.DataFrame, fit: bool, other_args):
        """This is a placeholder for a method that would be implemented to generate a particular feature from the data.

        Arguments:
        ---------
        data
            The data for which the feature is to be generated
        fit
            If true then we assume the provided data is training data and the method is allowed to learn any necessary
            values from the dataset. If false then we assume the provided data is validation or test data and the
            values necessary to generate the feature have been pre-computed from the training data in a previous step.
        other_args
            Include any additional arguments that the method may require.

        Returns:
        -------
        The original dataset with a new column (or columns) that contain(s) the new feature generated for each
        observation in the dataset.
        """

        return data

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
        transformed_data = self._example_feature1_method(train_data, fit=True, other_args=None)

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
        transformed_data = self._example_feature1_method(data, fit=False, other_args=None)

        # TODO: add in code below to transform datasets to generate other features as they are added

        return transformed_data


if __name__ == '__main__':

    # Imports
    from src.data_processor import DataProcessor

    # Load and clean the raw data
    myDP = DataProcessor()
    myDP.load_data(language='english', filepath='/put/data/directory/filepath/here')
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

