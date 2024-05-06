"""This script defines the class that houses the classification model(s) used to classify the HatEval data based on
the features generated from the FeatureEngineering class."""

# Libraries
import pandas as pd
import numpy as np


from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
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
        self.targets: Optional[List[str]] = None
        self.prediction_target = None

        # Initialize flag to indicate whether model training has occurred
        self.fitted = False

        # Create attribute to optionally store training data and predictions
        self.train_data: Optional[pd.DataFrame] = None
        self.train_est: Optional[pd.DataFrame] = None
        self.model_params: Optional[dict] = None

        # Create attributes for baseline model
        self.most_frequent_category: Optional[Dict[str, int]] = None

        # Create attributes for random forest classification model
        self.random_forest_classifier: Optional[RandomForestClassifier] = None
        self.features: Optional[List[str]] = None
        self.embedding_features: Optional[List[str]] = None


        # TODO: create necessary attributes for other models as they are added

    def _target_processing(self, data):
        """Processes the target categories into a uniform forman. So rather than having e.g. 3 binary categories for HS,
        TR and AG (with dependencies) we have a single 5 category problem (HS, HS+TR, HS+AG, HS+TR+AG, None).

        Arguments:
        ----------
        data
            The dataset, with the target task columns.

        Returns:
        --------
        Original dataframe with a new column containing the new target category.
        """

        # Specify which columns contain the target class(es) and order them
        tasks = [self.target_map[t] for t in self.tasks]
        task_cols = []
        if 'HS' in tasks:
            task_cols.append('HS')
        if 'TR' in tasks:
            task_cols.append('TR')
        if 'AG' in tasks:
            task_cols.append('AG')
        self.targets = task_cols

        # Specify classification objective
        target_categories = []
        for index, row in data.iterrows():
            targets = row[task_cols]
            # Specify if class is HS or not
            new_category = ''
            if targets.iloc[0] == 1:
                new_category = f'{task_cols[0]}+'
                # If class is HS, is it TR?
                if targets.iloc[1] == 1:
                    new_category = f'{new_category}{task_cols[1]}+'
                # If class is HS, is it AG?
                if targets.iloc[2] == 1:
                    new_category = f'{new_category}{task_cols[1]}'
                # Remove trailing +, if it exists
                new_category = new_category.rstrip('+')
            else:
                new_category = 'None'
            target_categories.append(new_category)

        data['Target'] = target_categories

        return data


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

    def _fit_random_forest_model(self, train_data: pd.DataFrame, features: Optional[List[str]],
                                 embedding_features: Optional[List[str]]) -> pd.DataFrame:
        """Trains a random forest model to predict the target category/categories specified in the tasks list.

        Arguments:
        ----------
        train_data
            The data set, with the complete set of engineered features, that is used to train the model.
        features
            The set of features to be used in the classification task.
        embedding_features
            The set of embedding-type features to be used in the classification task.


        Returns:
        --------
        A copy of the original dataframe with new columns appended that contain the random forest classifier predictions
        for the specified training task(s).
        """

        # At least one of features or embedding_features must be non-empty
        assert len(features) > 0 or len(embedding_features) > 0, \
            'At least one feature must be provided in order to train a classification model'

        # Specify which columns contain the target class(es)
        task_cols = [self.target_map[t] for t in self.tasks]
        self.task_cols = task_cols

        # Limit training data to include only the features and rid of NAs
        x_train = train_data[features]

        # Process embedding(s) features, if they exist
        self.embedding_features = embedding_features
        if embedding_features is not None:
            for ef in embedding_features:
                embeddings = np.stack(train_data[ef])
                col_prefix = f'{ef}_dim_'
                emb_cols = [col_prefix + str(dim) for dim in range(embeddings.shape[1])]
                embeddings = pd.DataFrame(embeddings, columns=emb_cols, index=train_data.index)
                x_train = pd.concat([x_train, embeddings], axis=1)

        # Specify classification objective(s)
        # If treating classification tasks separately as binary objectives
        if self.prediction_target == 'separate':
            y_train = train_data[self.task_cols]
            if y_train.shape[1] < 2:
                y_train = y_train.values.flatten()

            # Initialize Random Forest Classifier
            clf = RandomForestClassifier(n_estimators=self.model_params['n_estimators'],
                                         criterion=self.model_params['criterion'],
                                         max_depth=self.model_params['max_depth'],
                                         min_samples_split=self.model_params['min_samples_split'],
                                         min_samples_leaf=self.model_params['min_samples_leaf'],
                                         max_features=self.model_params['max_features'],
                                         bootstrap=self.model_params['bootstrap'],
                                         n_jobs=self.model_params['n_jobs'],
                                         random_state=self.model_params['random_state'],
                                         class_weight=self.model_params['class_weight'],
                                         max_samples=self.model_params['max_samples'])

            # Fit to the training data
            clf.fit(x_train, y_train)

            # Save the fit model
            self.random_forest_classifier = clf

            # Get the training data predictions
            y_pred = clf.predict(x_train)
            y_pred = pd.DataFrame(y_pred, columns=self.task_cols)
            pred_df = deepcopy(train_data)
            n_cols = len(pred_df.columns)
            for t in task_cols:
                pred_df.insert(loc=n_cols, column=f'{t}_prediction', value=y_pred[t].values)
                n_cols += 1

        # If treating classification tasks as a single, multi-class objective
        elif self.prediction_target == 'together':

            y_train = train_data['Target']

            # Initialize Random Forest Classifier
            clf = RandomForestClassifier(n_estimators=self.model_params['n_estimators'],
                                         criterion=self.model_params['criterion'],
                                         max_depth=self.model_params['max_depth'],
                                         min_samples_split=self.model_params['min_samples_split'],
                                         min_samples_leaf=self.model_params['min_samples_leaf'],
                                         max_features=self.model_params['max_features'],
                                         bootstrap=self.model_params['bootstrap'],
                                         n_jobs=self.model_params['n_jobs'],
                                         random_state=self.model_params['random_state'],
                                         class_weight=self.model_params['class_weight'],
                                         max_samples=self.model_params['max_samples'])

            # Fit to the training data
            clf.fit(x_train, y_train)

            # Save the fit model
            self.random_forest_classifier = clf

            # Get the training data predictions
            y_pred = clf.predict(x_train)
            y_pred = pd.DataFrame(y_pred, columns=['Target'])
            pred_df = deepcopy(train_data)
            target_preds = {f'{t}_prediction': [] for t in self.targets}
            for index, row in y_pred.iterrows():
                if row['Target'] == 'HS+TR+AG':
                    target_preds['HS_prediction'].append(1)
                    target_preds['TR_prediction'].append(1)
                    target_preds['AG_prediction'].append(1)
                elif row['Target'] == 'HS+TR':
                    target_preds['HS_prediction'].append(1)
                    target_preds['TR_prediction'].append(1)
                    target_preds['AG_prediction'].append(0)
                elif row['Target'] == 'HS+AG':
                    target_preds['HS_prediction'].append(1)
                    target_preds['TR_prediction'].append(0)
                    target_preds['AG_prediction'].append(1)
                elif row['Target'] == 'HS':
                    target_preds['HS_prediction'].append(1)
                    target_preds['TR_prediction'].append(0)
                    target_preds['AG_prediction'].append(0)
                else:
                    target_preds['HS_prediction'].append(0)
                    target_preds['TR_prediction'].append(0)
                    target_preds['AG_prediction'].append(0)
            n_cols = len(pred_df.columns)
            for k in target_preds.keys():
                pred_df.insert(loc=n_cols, column=k, value=target_preds[k])
                n_cols += 1

        return pred_df


    def _fit_svm_model(self, train_data: pd.DataFrame, features: Optional[List[str]],
                        embedding_features: Optional[List[str]]) -> pd.DataFrame:
        """Trains a support vector machine to predict the target category/categories specified in the tasks list.

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
        embedding_features
            The set of embedding-type features to be used in the classification task.


        Returns:
        --------
        A copy of the original dataframe with new columns appended that contain the random forest classifier predictions
        for the specified training task(s).
        """

        
        # At least one of features or embedding_features must be non-empty
        assert len(features) > 0 or len(embedding_features) > 0, \
            'At least one feature must be provided in order to train a classification model'

        # Identify all feature columns
        X_features = train_data[features].values if features else None

        # Turn embedding_features into one-dimensional features
        # This is done because the SVM cannot handle features of different dimensions.
        if embedding_features:

            embedding_ft_stats = []
            for feature in embedding_features:
                feature_array = train_data[feature].apply(np.array)

                # Calculate features for each embedding
                feature_means = feature_array.apply(np.mean)
                # feature_medians = feature_array.apply(np.median)
                feature_stdevs = feature_array.apply(np.std)
                # feature_skewness = feature_array.apply(lambda x: pd.Series(x).skew())
                # feature_kurtosis = feature_array.apply(lambda x: pd.Series(x).kurtosis())

                # Combine stats into a single matrix
                feature_stats = np.column_stack((feature_means, feature_stdevs))
                embedding_ft_stats.append(feature_stats)

        X_embedding_features = np.hstack(embedding_ft_stats)


        # Save model features
        self.features = features
        self.embedding_features = embedding_features

        # Concatenate features based on which are present:
        if X_features is not None and X_embedding_features is not None:
            X_ft = np.column_stack((X_features, X_embedding_features))                
        elif X_embedding_features is not None:
            X_ft = X_embedding_features
        else:
            X_ft = X_features


        # Specify which columns contain the target class(es)
        task_cols = [self.target_map[t] for t in self.tasks]

        self.task_cols = task_cols

        # If treating classification tasks separately as binary objectives
        if self.prediction_target == 'separate':
            # Combine target columns into one column if multiple tasks are given
            y = train_data[task_cols].values

            # Train SVM model
            clf = SVC(kernel=self.model_params['kernel'],
                      degree=self.model_params['degree'],
                      C=self.model_params['C'],
                      coef0=self.model_params['coef0'],
                      probability=self.model_params['probability'])
            multi_target_clf = MultiOutputClassifier(clf)
            multi_target_clf.fit(X_ft, y)

            # Save the fit model
            self.multi_target_classifier = multi_target_clf


            # Generate predictions on training data
            y_pred = multi_target_clf.predict(X_ft)
            # y_pred = pd.DataFrame(y_pred, columns=task_cols)

            # Create a DataFrame for predictions
            pred_df = deepcopy(train_data)
            for i, col in enumerate(task_cols):
                pred_df[f'{col}_prediction'] = y_pred[:, i]

            return pred_df

        # If treating classification tasks as a single, multi-class objective
        elif self.prediction_target == 'together':
            # Combine target columns into one column if multiple tasks are given
            y = train_data['Target'].values

            # Train SVM model
            clf = SVC(kernel=self.model_params['kernel'],
                      degree=self.model_params['degree'],
                      C=self.model_params['C'],
                      coef0=self.model_params['coef0'],
                      probability=self.model_params['probability'])
            clf.fit(X_ft, y)

            # Save the fit model
            self.multi_target_classifier = clf

            # Generate predictions on training data
            y_pred = clf.predict(X_ft)
            #
            y_pred = pd.DataFrame(y_pred, columns=['Target'])
            pred_df = deepcopy(train_data)
            target_preds = {f'{t}_prediction': [] for t in self.targets}
            for index, row in y_pred.iterrows():
                if row['Target'] == 'HS+TR+AG':
                    target_preds['HS_prediction'].append(1)
                    target_preds['TR_prediction'].append(1)
                    target_preds['AG_prediction'].append(1)
                elif row['Target'] == 'HS+TR':
                    target_preds['HS_prediction'].append(1)
                    target_preds['TR_prediction'].append(1)
                    target_preds['AG_prediction'].append(0)
                elif row['Target'] == 'HS+AG':
                    target_preds['HS_prediction'].append(1)
                    target_preds['TR_prediction'].append(0)
                    target_preds['AG_prediction'].append(1)
                elif row['Target'] == 'HS':
                    target_preds['HS_prediction'].append(1)
                    target_preds['TR_prediction'].append(0)
                    target_preds['AG_prediction'].append(0)
                else:
                    target_preds['HS_prediction'].append(0)
                    target_preds['TR_prediction'].append(0)
                    target_preds['AG_prediction'].append(0)
            n_cols = len(pred_df.columns)
            for k in target_preds.keys():
                pred_df.insert(loc=n_cols, column=k, value=target_preds[k])
                n_cols += 1

            return pred_df


    def fit(self, train_data: pd.DataFrame, tasks: List[str], prediction_target: str, keep_training_data: bool = True,
            parameters: Optional[dict] = None, features: Optional[List[str]] = None,
            embedding_features: Optional[List[str]] = None) -> pd.DataFrame:
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
        prediction_target
            If "separate" then the tasks are treated as separate classification problems. If "together" then the tasks
            are all grouped together and treated as a multi-class classification problem. This enables us to cut down on
            the number of class-combinations that are predicted, as we know that TR and AG values cannot be 1 without
            HS being 1.
        keep_training_data
            A boolean that indicates whether the training data should be stored on the model.
        parameters
            The dictionary of parameter values used to specify the model.
        features
            The list of features to be used in the classification task.
        embedding_features
            The set of embedding-type features to be used in the classification task.


        Returns:
        --------
        A copy of the original dataframe with new columns appended that contain the model predictions for the specified
        training task(s).

        """

        assert self.model_type is not None, 'The model_type must be specified in order to train a model.'

        # Save task list
        self.tasks = tasks

        # Process target tasks into single categorization task
        self.prediction_target = prediction_target
        train_data = self._target_processing(train_data)

        # Fit and predict baseline model
        if self.model_type == 'baseline':
            if keep_training_data:
                self.train_data = train_data
            pred_df = self._fit_baseline_model(train_data, tasks)

        # Fit and predict random forest classifier
        if self.model_type == 'random_forest':

            # Save the model features
            assert features is not None or embedding_features is not None, \
                'At least one feature must be provided in order to tran a Random Forest classification model.'
            self.features = features

            # Save the default model parameters
            self.model_params = {'n_estimators': 400, 'criterion': 'entropy', 'max_depth': None,
                                 'min_samples_split': 0.1, 'min_samples_leaf': 3, 'max_features': 'sqrt',
                                 'bootstrap': True, 'n_jobs': None, 'random_state': 42,
                                 'class_weight': 'balanced_subsample', 'max_samples': 0.2}
            
            # Replace specified defaults and save the provided model parameters
            if parameters is not None:
                for p in parameters.keys():
                    self.model_params[p] = parameters[p]

            # Save the training data
            if keep_training_data:
                self.train_data = train_data

            # Train the model
            pred_df = self._fit_random_forest_model(train_data, features, embedding_features)

        # Fit and predict SVM Classifier
        elif self.model_type == 'svm':

            # Save the model features
            assert features is not None or embedding_features is not None, \
                'At least one feature must be provided in order to train a Support Vector Machine classification model.'
            
            # Save the default model parameters
            self.model_params = {'kernel': 'poly', 'degree': 3, 'C': 1.0, 'coef0': 0, 'probability': True}

            # Replace specified defaults and save the provided model parameters
            if parameters is not None:
                for p in parameters.keys():
                    self.model_params[p] = parameters[p]

            # Save the default model parameters
            # highest performance hyperparameter setup (for separaate models) after some tuning
            self.model_params = {'kernel': 'poly', 'degree': 3, 'C': 1.0, 'coef0': 0, 'probability': True}

            # Replace specified defaults and save the provided model parameters
            if parameters is not None:
                for p in parameters.keys():
                    self.model_params[p] = parameters[p]

            # train the classifiers
            pred_df = self._fit_svm_model(train_data, features, embedding_features)

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

            # Add appropriate engineered features
            x_data = data[self.features]

            # Add appropriate embedding features
            if self.embedding_features is not None:
                for ef in self.embedding_features:
                    embeddings = np.stack(data[ef])
                    col_prefix = f'{ef}_dim_'
                    emb_cols = [col_prefix + str(dim) for dim in range(embeddings.shape[1])]
                    embeddings = pd.DataFrame(embeddings, columns=emb_cols, index=data.index)
                    x_data = pd.concat([x_data, embeddings], axis=1)

            # Get the predictions
            if self.prediction_target == 'separate':
                y_pred = self.random_forest_classifier.predict(x_data)
                y_pred = pd.DataFrame(y_pred, columns=self.task_cols)
                # pred_df = deepcopy(x_data)
                n_cols = len(pred_df.columns)
                for t in self.task_cols:
                    pred_df.insert(loc=n_cols, column=f'{t}_prediction', value=y_pred[t].values)
                    n_cols += 1

            elif self.prediction_target == 'together':
                y_pred = self.random_forest_classifier.predict(x_data)
                y_pred = pd.DataFrame(y_pred, columns=['Target'])
                pred_df = deepcopy(data)
                target_preds = {f'{t}_prediction': [] for t in self.targets}
                for index, row in y_pred.iterrows():
                    if row['Target'] == 'HS+TR+AG':
                        target_preds['HS_prediction'].append(1)
                        target_preds['TR_prediction'].append(1)
                        target_preds['AG_prediction'].append(1)
                    elif row['Target'] == 'HS+TR':
                        target_preds['HS_prediction'].append(1)
                        target_preds['TR_prediction'].append(1)
                        target_preds['AG_prediction'].append(0)
                    elif row['Target'] == 'HS+AG':
                        target_preds['HS_prediction'].append(1)
                        target_preds['TR_prediction'].append(0)
                        target_preds['AG_prediction'].append(1)
                    elif row['Target'] == 'HS':
                        target_preds['HS_prediction'].append(1)
                        target_preds['TR_prediction'].append(0)
                        target_preds['AG_prediction'].append(0)
                    else:
                        target_preds['HS_prediction'].append(0)
                        target_preds['TR_prediction'].append(0)
                        target_preds['AG_prediction'].append(0)
                n_cols = len(pred_df.columns)
                for k in target_preds.keys():
                    pred_df.insert(loc=n_cols, column=k, value=target_preds[k])
                    n_cols += 1

        if self.model_type == 'svm':
            # Identify all feature columns
            X_features = data[self.features].values if self.features else None

            # Turn embedding_features into one-dimensional features
            # This is done because the SVM cannot handle features of different dimensions.
            if self.embedding_features:

                embedding_ft_stats = []
                for feature in self.embedding_features:
                    feature_array = data[feature].apply(np.array)

                    # Calculate features for each embedding
                    feature_means = feature_array.apply(np.mean)
                    # feature_medians = feature_array.apply(np.median)
                    feature_stdevs = feature_array.apply(np.std)
                    # feature_skewness = feature_array.apply(lambda x: pd.Series(x).skew())
                    # feature_kurtosis = feature_array.apply(lambda x: pd.Series(x).kurtosis())

                    # Combine stats into a single matrix
                    feature_stats = np.column_stack((feature_means, feature_stdevs))
                    embedding_ft_stats.append(feature_stats)


            X_embedding_features = np.hstack(embedding_ft_stats)

            # Concatenate features based on which are present:
            if X_features is not None and X_embedding_features is not None:
                X_ft = np.column_stack((X_features, X_embedding_features))                
            elif X_embedding_features is not None:
                X_ft = X_embedding_features
            else:
                X_ft = X_features


            # Generate predictions on training data
            if self.prediction_target == 'separate':
                y_pred = self.multi_target_classifier.predict(X_ft)
                # y_pred = pd.DataFrame(y_pred, columns=self.task_cols)

                # Create a DataFrame for predictions
                pred_df = deepcopy(data)
                for i, col in enumerate(self.task_cols):
                    pred_df[f'{col}_prediction'] = y_pred[:, i]

            elif self.prediction_target == 'together':
                y_pred = self.multi_target_classifier.predict(X_ft)
                #
                y_pred = pd.DataFrame(y_pred, columns=['Target'])
                pred_df = deepcopy(data)
                target_preds = {f'{t}_prediction': [] for t in self.targets}
                for index, row in y_pred.iterrows():
                    if row['Target'] == 'HS+TR+AG':
                        target_preds['HS_prediction'].append(1)
                        target_preds['TR_prediction'].append(1)
                        target_preds['AG_prediction'].append(1)
                    elif row['Target'] == 'HS+TR':
                        target_preds['HS_prediction'].append(1)
                        target_preds['TR_prediction'].append(1)
                        target_preds['AG_prediction'].append(0)
                    elif row['Target'] == 'HS+AG':
                        target_preds['HS_prediction'].append(1)
                        target_preds['TR_prediction'].append(0)
                        target_preds['AG_prediction'].append(1)
                    elif row['Target'] == 'HS':
                        target_preds['HS_prediction'].append(1)
                        target_preds['TR_prediction'].append(0)
                        target_preds['AG_prediction'].append(0)
                    else:
                        target_preds['HS_prediction'].append(0)
                        target_preds['TR_prediction'].append(0)
                        target_preds['AG_prediction'].append(0)
                n_cols = len(pred_df.columns)
                for k in target_preds.keys():
                    pred_df.insert(loc=n_cols, column=k, value=target_preds[k])
                    n_cols += 1

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
    train_df = myFE.fit_transform(myDP.processed_data['train'], embedding_file_path='../data/glove.twitter.27B.25d.txt',
                                  embedding_dim=25)
    val_df = myFE.transform(myDP.processed_data['validation'])

    # Instantiate the model
    myClassifier = ClassificationModel('random_forest')

    # Train the model
    features = ['percent_capitals', '!_count_normalized', '?_count_normalized', '$_count_normalized',
                '*_count_normalized', 'negative', 'positive', 'anger', 'anticipation', 'disgust', 'fear', 'joy',
                'sadness', 'surprise', 'trust']
    train_pred = myClassifier.fit(train_df,
                                  tasks=['hate_speech_detection', 'target_or_general', 'aggression_detection'],
                                  keep_training_data=False, features=features)

    # Run the model on the validation data
    val_pred = myClassifier.predict(val_df)

    # View a sample of the results
    train_df.head()
    val_df.head()
