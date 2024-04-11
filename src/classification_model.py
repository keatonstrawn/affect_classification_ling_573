"""This script defines the class that houses the classification model(s) used to classify the HatEval data based on
the features generated from the FeatureEngineering class."""

# Libraries
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional, Dict
from copy import deepcopy


# Define class to house classification models
class ClassificationModel:

    def __init__(self, model_type: str):
        """Generates features from processed data to be used in hate speech detection tasks A and B, as specified in
        SemEval 2019 task 5.

        Includes methods to train the following classification models:
            * baseline model: Guesses the target class most commonly seen in the training data.
            * random forest: Aggregates probabilistic estimates from decision trees fit on subsets of the dataset to
                predict the target class. (Note: The probability aggregation is not the same as the approach that allows
                each tree to 'vote' for a single class.)
            * example model3 --> fill this in with model info

            Parameters
            ----------
            model_type
                Specifies which type of model is used to perform classification. Options include:
                    'baseline'
                    'random_forest'
                    'model3'
        """

        # Initialize model parameters
        self.model_type = model_type
        self.model_objectives = []
        self.tasks: Optional[List[str]] = None
        self.target_map = {'hate_speech_detection': 'HS', 'target_or_general': 'TR', 'aggression_detection': 'AG'}

        # Initialize flag to indicate whether model training has occurred
        self.fitted = False

        # Create attribute to optionally store training data and predictions
        self.train_data: Optional[pd.DataFrame] = None
        self.train_est: Optional[pd.DataFrame] = None

        # Create attributes for baseline model
        self.most_frequent_category: Optional[Dict[str, int]] = None

        # Create attributes for random forest classification model
        self.random_forest_classifier: Optional[RandomForestClassifier] = None
        self.features: Optional[List[str]] = None


        # TODO: create necessary attributes for other models as they are added

    def _fit_baseline_model(self, train_data: pd.DataFrame, tasks: List[str]) -> pd.DataFrame:
        """Trains a baseline categorization model, which predicts the target category most frequently seen in the
        training data for every

        Arguments:
        ----------
        train_data
            The data set, with the complete set of engineered features, that is used to train the model.
        tasks
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
        pred_df = train_data

        # Get prediction value(s)
        most_frequent_target = {}
        for t in tasks:
            if t == 'hate_speech_detection':
                hs_mode = train_data['HS'].mode().values[0]
                most_frequent_target['HS'] = hs_mode
            if t == 'target_or_general':
                tr_mode = train_data['TR'].mode().values[0]
                most_frequent_target['TR'] = tr_mode
            if t == 'aggression_detection':
                ag_mode = train_data['AG'].mode().values[0]
                most_frequent_target['AG'] = ag_mode

        # Save trained predictions
        self.most_frequent_category = most_frequent_target

        # Add predictions to the dataframe
        n_cols = len(pred_df.columns)
        for k in most_frequent_target.keys():
            pred = most_frequent_target[k]
            pred_df.insert(loc=n_cols, column=f'{k}_prediction', value=pred)

        return pred_df

    def _fit_random_forest_model(self, train_data: pd.DataFrame, features: List[str], tasks: List[str]) -> pd.DataFrame:
        """Trains a random forest model to predict the target category/categories specified in the tasks list.

        Arguments:
        ----------
        train_data
            The data set, with the complete set of engineered features, that is used to train the model.
        features
            The set of features to be used in the classification task.
        tasks
            The classification task(s) that the model is being trained to predict. If a list of tasks is given then the
            same model is trained to predict each of those labels simultaneously. To train an individual model for each
            task, a new model must be instantiated and fit for each label. Task labels may include any of the following:
            'hate_speech_detection', 'target_or_general', 'aggression_detection'.


        Returns:
        --------
        A copy of the original dataframe with new columns appended that contain the random forest classifier predictions
        for the specified training task(s).
        """

        # Specify which columns contain the target class(es)
        task_cols = [self.target_map[t] for t in tasks]

        # Limit training data to include only the features and rid of NAs
        x_train = train_data[features]

        # Specify classification objective(s)
        y_train = train_data[task_cols]
        if y_train.shape[1] < 2:
            y_train = y_train.values.flatten()

        # Initialize Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=400, criterion='entropy', max_depth=None, min_samples_split=0.1,
                                     min_samples_leaf=3, max_features='sqrt', bootstrap=True, n_jobs=None,
                                     random_state=42, class_weight='balanced_subsample', max_samples=0.2)

        # Fit to the training data
        clf.fit(x_train, y_train)

        # Save the fit model
        self.random_forest_classifier = clf

        # Get the training data predictions
        y_pred = clf.predict(x_train)
        y_pred = pd.DataFrame(y_pred, columns=task_cols)
        pred_df = deepcopy(train_data)
        n_cols = len(pred_df.columns)
        for t in task_cols:
            pred_df.insert(loc=n_cols, column=f'{t}_prediction', value=y_pred[t].values)
            n_cols += 1

        return pred_df

    def fit(self, train_data: pd.DataFrame, tasks: List[str], keep_training_data: bool = True,
            features: Optional[List[str]] = None) -> pd.DataFrame:
        """Uses the provided data to train the model to perform the specified classification task.

        Arguments:
        ----------
        train_data
            The data set, with the complete set of engineered features, that is used to train the model.
        tasks
            The classification task(s) that the model is being trained to predict. If a list of tasks is given then the
            same model is trained to predict each of those labels simultaneously. To train an individual model for each
            task, a new model must be instantiated and fit for each label. Task labels may include any of the following:
            'hate_speech_detection', 'target_or_general', 'aggression_detection'.
        keep_training_data
            A boolean that indicates whether the training data should be stored on the model.
        features
            The list of features to be used in the classification task.


        Returns:
        --------
        A copy of the original dataframe with new columns appended that contain the model predictions for the specified
        training task(s).

        """

        assert self.model_type is not None, 'The model_type must be specified in order to train a model.'

        # Save task list
        self.tasks = tasks

        # Fit and predict baseline model
        if self.model_type == 'baseline':
            if keep_training_data:
                self.train_data = train_data
            pred_df = self._fit_baseline_model(train_data, tasks)

        # Fit and predict random forest classifier
        if self.model_type == 'random_forest':
            assert features is not None, \
                'The feature list must be provided in order to tran a Random Forest classification model.'
            self.features = features
            if keep_training_data:
                self.train_data = train_data
            pred_df = self._fit_random_forest_model(train_data, features, tasks)

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

        if self.model_type == 'random_forest':

            # Clean the data
            x_data = data[self.features]

            # Get the predictions
            task_cols = [self.target_map[t] for t in self.tasks]
            y_pred = self.random_forest_classifier.predict(x_data)
            y_pred = pd.DataFrame(y_pred, columns=task_cols)
            n_cols = len(pred_df.columns)
            for t in task_cols:
                pred_df.insert(loc=n_cols, column=f'{t}_prediction', value=y_pred[t].values)
                n_cols += 1

        # TODO: add sections below that perform prediction for other classifier models as they are added

        return pred_df


if __name__ == '__main__':

    # Imports
    from src.data_processor import DataProcessor
    from src.feature_engineering import FeatureEngineering

    # Load and clean the raw data
    myDP = DataProcessor()
    myDP.load_data(language='english', filepath='../data')  # May need to change to './data' or 'data' if on a Mac
    myDP.clean_data()

    # Generate the features for model training
    myFE = FeatureEngineering()
    train_df = myFE.fit_transform(myDP.processed_data['train'])
    val_df = myFE.transform(myDP.processed_data['validation'])

    # Instantiate the model
    myClassifier = ClassificationModel('random_forest')

    # Train the model
    features = ['percent_capitals', '!_count', '?_count', '$_count', '*_count', 'negative', 'positive', 'anger',
                'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    train_pred = myClassifier.fit(train_df,
                                  tasks=['hate_speech_detection', 'target_or_general', 'aggression_detection'],
                                  keep_training_data=False, features=features)

    # Run the model on the validation data
    val_pred = myClassifier.predict(val_df)

    # View a sample of the results
    train_df.head()
    val_df.head()




