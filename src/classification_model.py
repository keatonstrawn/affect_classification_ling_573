"""This script defines the class that houses the classification model(s) used to classify the HatEval data based on
the features generated from the FeatureEngineering class."""

# Libraries
import pandas as pd

from sklearn import svm
from typing import List, Optional, Dict
from copy import deepcopy

# Define class to house classification models
class ClassificationModel:

    def __init__(self, model_type: str):
        """ Description
        """

        # Initialize model parameters
        self.model_type = model_type
        self.model_objectives = []

        # Initialize flag to indicate whether model training has occurred
        self.fitted = False

        # Create attributes to optionally store training data
        self.train_data: Optional[pd.DataFrame] = None

        # Create attributes for baseline model
        self.most_frequent_category: Optional[Dict[str, int]] = None

    def _fit_baseline_model(self, train_data: pd.DataFrame, task: List[str]) -> pd.DataFrame:
        """Trains a baseline categorization model, which predicts the target category most frequently seen in the
        training data for every

        Arguments:
        ----------
        train_data
            The data set, with the complete set of engineered features, that is used to train the model.
        task
            The classification task(s) that the model is being trained to predict. If a list of tasks is given then the
            same model is trained to predict each of those labels simultaneously. To train an individual model for each
            task, a new model must be instantiated and fit for each label. Task labels may include any of the following:
            'hate_speech_detection', 'target_or_general', 'aggression_detection'.


        Returns:
        --------
        A copy of the original dataframe with new columns appended that contain the baseline model predictions for the
        specified training task(s).
        """

        # Initialise prediction dataframe
        pred_df = deepcopy(train_data)

        # Get prediction value(s)
        most_frequent_target = {}
        for t in task:
            if t == 'hate_speech_detection':
                hs_mode = train_data['HS'].mode()
                most_frequent_target['HS'] = hs_mode
            if t == 'target_or_general':
                tr_mode = train_data['TR'].mode()
                most_frequent_target['TR'] = tr_mode
            if t == 'aggression_detection':
                ag_mode = train_data['AG'].mode()
                most_frequent_target['AG'] = ag_mode

        # Save trained predictions
        self.most_frequent_category = most_frequent_target

        # Add predictions to the dataframe
        n_cols = len(pred_df.columns)
        for k in most_frequent_target:
            pred = most_frequent_target[k]
            pred_df.insert(loc=n_cols, column=f'{k}_prediction', value=pred)

        return pred_df

    def fit(self, train_data: pd.DataFrame, task: List[str], keep_training_data: bool = True) -> pd.DataFrame:
        """Uses the provided data to train the model to perform the specified classification task.

        Arguments:
        ----------
        train_data
            The data set, with the complete set of engineered features, that is used to train the model.
        task
            The classification task(s) that the model is being trained to predict. If a list of tasks is given then the
            same model is trained to predict each of those labels simultaneously. To train an individual model for each
            task, a new model must be instantiated and fit for each label. Task labels may include any of the following:
            'hate_speech_detection', 'target_or_general', 'aggression_detection'.
        keep_training_data
            A boolean that indicates whether the training data should be stored on the model.


        Returns:
        --------
        A copy of the original dataframe with new columns appended that contain the model predictions for the specified
        training task(s).

        """

        assert self.model_type is not None, 'The model_type must be specified in order to train a model.'

        # Fit and predict baseline model
        if self.model_type == 'baseline':
            if keep_training_data:
                self.train_data = train_data
            pred_df = self._fit_baseline_model(train_data, task)

        # TODO: Add other sections below that reference helper methods to do any necessary processing and fit_predict
        #  for other classifier models as they are added

        # Flag that model fitting has occurred
        self.fitted = True

        return pred_df

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Description

        Arguments:
        ----------
        data
            The data set, with the complete set of engineered features, that we want to make predictions for using a
            trained model.

        Returns:
        --------
        A copy of the original dataframe with new columns appended that contain the model predictions.
        """

        assert self.fitted, 'You must train a model before calling the predict method.'

        # Initialise prediction dataframe
        pred_df = deepcopy(data)

        # Predictions for baseline model
        if self.model_type == 'baseline':

            # Add predictions to the dataframe
            n_cols = len(pred_df.columns)
            for k in self.most_frequent_category:
                pred = self.most_frequent_category[k]
                pred_df.insert(loc=n_cols, column=f'{k}_prediction', value=pred)

        # TODO: add sections below that perform prediction for other classifier models as they are added

        return pred_df


if __name__ == '__main__':

    # Imports
    from src.data_processor import DataProcessor
    from src.feature_engineering import FeatureEngineering

    # Load and clean the raw data
    myDP = DataProcessor()
    myDP.load_data(language='english', filepath='/put/data/directory/filepath/here')
    myDP.clean_data()

    # Generate the features for model training
    myFE = FeatureEngineering()
    train_df = myFE.fit_transform(myDP.processed_data['train'])
    val_df = myFE.transform(myDP.processed_data['validation'])

    # Instantiate the model
    myClassifier = ClassificationModel('baseline')

    # Train the model
    train_pred = myClassifier.fit(train_df, task='hate_speech_detection', keep_training_data=False)

    # Run the model on the validation data
    val_pred = myClassifier.predict(val_df)

    # View a sample of the results
    train_df.head()
    val_df.head()
