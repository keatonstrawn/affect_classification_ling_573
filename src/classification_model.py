"""This script defines the class that houses the classification model(s) used to classify the HatEval data based on
the features generated from the FeatureEngineering class."""

# Libraries
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
                    'svm'
                    'logistic_regression'
                    'ensemble_lr'
                    'ensemble_dt'
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

        # Create attributes for SVM classification model
        self.multi_target_classifier: Optional[MultiOutputClassifier] = None

        # Create attributes for logistic regression classification model
        self.logistic_regression_classifier: Optional[LogisticRegression] = None

        # Create attributes for ensembling models
        self.ensembler_models: Optional[Dict[str: any]] = {}
        self.submodel_results: Optional[pd.DataFrame] = None
        self.submodel_results_test: Optional[pd.DataFrame] = None

        # TODO: create necessary attributes for other models as they are added

    def _target_processing(self, data):
        """Processes the target categories into a uniform format. So rather than having e.g. 3 binary categories for HS,
        TR and AG (with dependencies) we have a single 5 category problem (HS, HS+TR, HS+AG, HS+TR+AG, NotHS).

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
                    new_category = f'{new_category}{task_cols[2]}'
                # Remove trailing +, if it exists
                new_category = new_category.rstrip('+')
            else:
                new_category = 'NotHS'
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
        embedding_features
            The set of embedding-type features to be used in the classification task.


        Returns:
        --------
        A copy of the original dataframe with new columns appended that contain the SVM classifier predictions
        for the specified training task(s).
        """

        # At least one of features or embedding_features must be non-empty
        assert len(features) > 0 or len(embedding_features) > 0, \
            'At least one feature must be provided in order to train an SVM classification model'

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

        # Save model features
        self.features = features
        self.embedding_features = embedding_features

        # If treating classification tasks separately as binary objectives
        if self.prediction_target == 'separate':
            # Combine target columns into one column if multiple tasks are given
            y_train = train_data[task_cols].values

            # Train SVM model
            clf = SVC(kernel=self.model_params['kernel'],
                      degree=self.model_params['degree'],
                      C=self.model_params['C'],
                      coef0=self.model_params['coef0'],
                      probability=self.model_params['probability'])
            multi_target_clf = MultiOutputClassifier(clf)
            multi_target_clf.fit(x_train, y_train)

            # Save the fit model
            self.multi_target_classifier = multi_target_clf

            # Generate predictions on training data
            y_pred = multi_target_clf.predict(x_train)

            # Create a DataFrame for predictions
            pred_df = deepcopy(train_data)
            for i, col in enumerate(task_cols):
                pred_df[f'{col}_prediction'] = y_pred[:, i]

            return pred_df

        # If treating classification tasks as a single, multi-class objective
        elif self.prediction_target == 'together':
            # Combine target columns into one column if multiple tasks are given
            y_train = train_data['Target'].values

            # Train SVM model
            clf = SVC(kernel=self.model_params['kernel'],
                      degree=self.model_params['degree'],
                      C=self.model_params['C'],
                      coef0=self.model_params['coef0'],
                      probability=self.model_params['probability'])
            clf.fit(x_train, y_train)

            # Save the fit model
            self.multi_target_classifier = clf

            # Generate predictions on training data
            y_pred = clf.predict(x_train)
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

    def _fit_logistic_regression_model(self, train_data: pd.DataFrame, features: Optional[List[str]],
                       embedding_features: Optional[List[str]]) -> pd.DataFrame:
        """Trains a support vector machine to predict the target category/categories specified in the tasks list.

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
        A copy of the original dataframe with new columns appended that contain the logistic regression predictions
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

        # Save model features
        self.features = features
        self.embedding_features = embedding_features

        # Specify which columns contain the target class(es)
        task_cols = [self.target_map[t] for t in self.tasks]

        self.task_cols = task_cols

        # If treating classification tasks separately as binary objectives
        if self.prediction_target == 'separate':
            # Combine target columns into one column if multiple tasks are given
            y_train = train_data[task_cols].values

            # Train logistic regression
            clf = LogisticRegression(penalty=self.model_params['penalty'],  # 'l2'
                                     random_state=self.model_params['random_state'],  # 42
                                     solver=self.model_params['solver'],  # 'sag'
                                     multi_class='ovr',
                                     max_iter=self.model_params['max_iter'])  # 1000
            clf.fit(x_train, y_train)

            # Save the fit model
            self.logistic_regression_classifier = clf

            # Generate predictions on training data
            y_pred = clf.predict(x_train)

            # Create a DataFrame for predictions
            pred_df = deepcopy(train_data)
            for i, col in enumerate(task_cols):
                pred_df[f'{col}_prediction'] = y_pred[:, i]

            return pred_df

        # If treating classification tasks as a single, multi-class objective
        elif self.prediction_target == 'together':
            # Combine target columns into one column if multiple tasks are given
            y_train = train_data['Target'].values

            # Train logistic regression
            clf = LogisticRegression(penalty=self.model_params['penalty'],
                                     random_state=self.model_params['random_state'],
                                     solver=self.model_params['solver'],
                                     multi_class='multinomial',
                                     max_iter=self.model_params['max_iter'])
            clf.fit(x_train, y_train)

            # Save the fit model
            self.logistic_regression_classifier = clf

            # Generate predictions on training data
            y_pred = clf.predict(x_train)
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


    def _fit_ensemble_model(self, train_data: pd.DataFrame, ensembler: str, features: Optional[List[str]],
                       embedding_features: Optional[List[str]]) -> pd.DataFrame:
        """Trains an ensembler to predict the target category/categories specified in the tasks list.

        The sub-models include a Support Vector Machine Regression, a Random Forest Regression, a Logistic Regression
        and a fine-tuned BERT model with an additional classification layer (so BERT + Linear Regression with Softmax).
        The results (provided as percent probabilities for each class) of each of these models is then provided to the
        ensemble model, which is either a logistic regression or a decision tree. (These classifiers were chosen due to
        their transparency; their design will make it very easy to determine which submodels are most informative under
        specific conditions. The ensembler looks at the provided results from the submodels to make the final
        predictions. Note that this predictor is set up so that it requires the problem to be set up as a 5-class target
        task, not 3 separate binary tasks.

        Arguments:
        ----------
        train_data
            The data set, with the complete set of engineered features, that is used to train the model.
        ensembler
            Specifies whether to use a Logistic Regression ('LR') or Decision Tree ('DT') as the ensembling classifier.
        features
            The set of features to be used in the classification task.
        embedding_features
            The set of embedding-type features to be used in the classification task.


        Returns:
        --------
        A copy of the original dataframe with new columns appended that contain each of the sub-model probabilities, as
        well as the overall ensembler predictions.
        """

        ##### Process the data #####

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

        # Save model features
        self.features = features
        self.embedding_features = embedding_features

        # Specify which columns contain the target class(es)
        task_cols = [self.target_map[t] for t in self.tasks]
        self.task_cols = task_cols

        # Combine target columns into one column if multiple tasks are given
        y_train = train_data['Target'].values

        ##### Fit the submodels and add their predictions to the DF #####
        ensemble_train_df = deepcopy(train_data[['Target']])
        ensemble_features = []

        # SVM
        # Train SVM model
        clf_svm = SVC(kernel=self.model_params['SVM']['kernel'],
                  degree=self.model_params['SVM']['degree'],
                  C=self.model_params['SVM']['C'],
                  coef0=self.model_params['SVM']['coef0'],
                  probability=True)
        clf_svm.fit(x_train, y_train)
        # Save the fit model
        self.ensembler_models['SVM'] = clf_svm
        # Generate predictions on training data
        y_pred_svm = clf_svm.predict_proba(x_train)
        # Add sub-model predictions to dataframe
        n = ensemble_train_df.shape[1]
        for i in range(y_pred_svm.shape[1]):
            col_title = f'SVM_{clf_svm.classes_[i]}'
            vals = y_pred_svm[:, i]
            ensemble_train_df.insert(loc=n, column=col_title, value=vals)
            ensemble_features.append(col_title)
            n += 1

        # Random Forest
        # Train the RF model
        clf_rf = RandomForestClassifier(n_estimators=self.model_params['random_forest']['n_estimators'],
                                     criterion=self.model_params['random_forest']['criterion'],
                                     max_depth=self.model_params['random_forest']['max_depth'],
                                     min_samples_split=self.model_params['random_forest']['min_samples_split'],
                                     min_samples_leaf=self.model_params['random_forest']['min_samples_leaf'],
                                     max_features=self.model_params['random_forest']['max_features'],
                                     bootstrap=self.model_params['random_forest']['bootstrap'],
                                     n_jobs=self.model_params['random_forest']['n_jobs'],
                                     random_state=self.model_params['random_forest']['random_state'],
                                     class_weight=self.model_params['random_forest']['class_weight'],
                                     max_samples=self.model_params['random_forest']['max_samples'])
        clf_rf.fit(x_train, y_train)
        # Save the fit model
        self.ensembler_models['random_forest_classifier'] = clf_rf
        # Generate predictions on training data
        y_pred_rf = clf_rf.predict_proba(x_train)
        # Add sub-model predictions to dataframe
        n = ensemble_train_df.shape[1]
        for i in range(y_pred_rf.shape[1]):
            col_title = f'RF_{clf_rf.classes_[i]}'
            vals = y_pred_rf[:, i]
            ensemble_train_df.insert(loc=n, column=col_title, value=vals)
            ensemble_features.append(col_title)
            n += 1

        # Logistic Regression
        # Train the LR model
        clf_lr = LogisticRegression(penalty=self.model_params['logistic_regression']['penalty'],
                                 random_state=self.model_params['logistic_regression']['random_state'],
                                 solver=self.model_params['logistic_regression']['solver'],
                                 multi_class='multinomial',
                                 max_iter=self.model_params['logistic_regression']['max_iter'])
        clf_lr.fit(x_train, y_train)
        # Save the fit model
        self.ensembler_models['logistic_regression_classifier'] = clf_lr
        # Generate predictions on training data
        y_pred_lr = clf_lr.predict_proba(x_train)
        # Add sub-model predictions to dataframe
        n = ensemble_train_df.shape[1]
        for i in range(y_pred_lr.shape[1]):
            col_title = f'LR_{clf_lr.classes_[i]}'
            vals = y_pred_lr[:, i]
            ensemble_train_df.insert(loc=n, column=col_title, value=vals)
            ensemble_features.append(col_title)
            n += 1


        # BERT
        # TODO: add reference to pre-trained BERT model call here

        ##### Fit the ensembler and provide it JUST the submodel predictions as input #####

        assert ensembler == 'LR' or ensembler == 'DT', \
               "ensembler must be specified as one of 'LR' or 'DT' in order to run."

        x_ensemble_train = ensemble_train_df[ensemble_features]
        y_ensemble_train = ensemble_train_df['Target']

        if ensembler == 'LR':
            # Train the LR Ensembler
            clf = LogisticRegression(penalty=self.model_params['ensembler']['penalty'],
                                        random_state=self.model_params['ensembler']['random_state'],
                                        solver=self.model_params['ensembler']['solver'],
                                        multi_class='multinomial',
                                        max_iter=self.model_params['ensembler']['max_iter'])
            clf.fit(x_ensemble_train, y_ensemble_train)
            # Save the fit model
            self.ensembler_models['ensembler'] = clf
            # Generate predictions on training data
            y_pred = clf.predict(x_ensemble_train)

        elif ensembler == 'DT':
            # Train the Decision Tree Ensembler
            clf = DecisionTreeClassifier(criterion=self.model_params['ensembler']['criterion'],
                                         splitter=self.model_params['ensembler']['splitter'],
                                         max_depth=None,
                                         min_samples_split=2,
                                         min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0,
                                         max_features=self.model_params['ensembler']['max_features'],
                                         random_state=self.model_params['ensembler']['random_state'],
                                         max_leaf_nodes=None,
                                         min_impurity_decrease=0.0,
                                         class_weight=self.model_params['ensembler']['class_weight'],
                                         ccp_alpha=self.model_params['ensembler']['ccp_alpha'])
            clf.fit(x_ensemble_train, y_ensemble_train)
            # Save the fit model
            self.ensembler_models['ensembler'] = clf
            # Generate predictions on training data
            y_pred = clf.predict(x_ensemble_train)

        # Add ensemble predictions to sub-model
        n = ensemble_train_df.shape[1]
        # for i in range(y_pred.shape[1]):
        #     col_title = f'Ensemble_{clf_lr.classes_[i]}'
        #     vals = y_pred[:, i]
        #     ensemble_train_df.insert(loc=n, column=col_title, value=vals)
        #     ensemble_features.append(col_title)
        #     n += 1
        ensemble_train_df.insert(loc=n, column='Ensemble_target', value=y_pred)
        # Save results
        self.submodel_results = deepcopy(ensemble_train_df)

        # Format predictions appropriately
        y_pred = pd.DataFrame(y_pred, index=x_ensemble_train.index, columns=['Target'])
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
            # highest performance hyperparameter setup (for separaate models) after some tuning
            self.model_params = {'kernel': 'poly', 'degree': 3, 'C': 1.0, 'coef0': 0, 'probability': True}

            # Replace specified defaults and save the provided model parameters
            if parameters is not None:
                for p in parameters.keys():
                    self.model_params[p] = parameters[p]

            # train the classifiers
            pred_df = self._fit_svm_model(train_data, features, embedding_features)

        elif self.model_type == 'logistic_regression':

            # Ensure we have model features
            assert features is not None or embedding_features is not None, \
                'At least one feature must be provided in order to train a Logistic Regression classification model.'

            # Save the default model parameters
            # TODO: NEED TO DO SOME HYPERPARAMETER TUNING FOR THIS MODEL
            self.model_params = {'penalty': 'l2', 'random_state': 42, 'solver': 'sag', 'max_iter': 1000}

            # Replace specified defaults and save the provided model parameters
            if parameters is not None:
                for p in parameters.keys():
                    self.model_params[p] = parameters[p]

            # train the classifiers
            pred_df = self._fit_logistic_regression_model(train_data, features, embedding_features)

        elif self.model_type == 'ensembler_lr':

            # Ensure we have model features
            assert features is not None or embedding_features is not None, \
                'At least one feature must be provided in order to train an Ensembler classification model.'
            self.features = features

            # Save the default model parameters
            # TODO: NEED TO DO SOME HYPERPARAMETER TUNING FOR THIS MODEL
            self.model_params = {}
            self.model_params['SVM'] = {'kernel': 'poly', 'degree': 3, 'C': 1.0, 'coef0': 0, 'probability': True}
            self.model_params['random_forest'] = {'n_estimators': 400, 'criterion': 'entropy', 'max_depth': None,
                'min_samples_split': 0.1, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'bootstrap': True,
                'n_jobs': None, 'random_state': 42, 'class_weight': 'balanced_subsample', 'max_samples': 0.2}
            self.model_params['logistic_regression'] = {'penalty': 'l2', 'random_state': 42, 'solver': 'sag',
                'max_iter': 1000}
            self.model_params['ensembler'] = {'penalty': 'l2', 'random_state': 42, 'solver': 'sag',
                'max_iter': 1000}

            # Replace specified defaults and save the provided model parameters
            if parameters is not None:
                for model in parameters.keys():
                    for p in parameters[model].keys():
                        self.model_params[model][p] = parameters[model][p]

            # train the classifiers
            pred_df = self._fit_ensemble_model(train_data, 'LR', features, embedding_features)

        elif self.model_type == 'ensembler_dt':

            # Ensure we have model features
            assert features is not None or embedding_features is not None, \
                'At least one feature must be provided in order to train an Ensembler classification model.'
            self.features = features

            # Save the default model parameters
            # TODO: NEED TO DO SOME HYPERPARAMETER TUNING FOR THIS MODEL
            self.model_params = {}
            self.model_params['SVM'] = {'kernel': 'poly', 'degree': 3, 'C': 1.0, 'coef0': 0, 'probability': True}
            self.model_params['random_forest'] = {'n_estimators': 400, 'criterion': 'entropy', 'max_depth': None,
                'min_samples_split': 0.1, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'bootstrap': True,
                'n_jobs': None, 'random_state': 42, 'class_weight': 'balanced_subsample', 'max_samples': 0.2}
            self.model_params['logistic_regression'] = {'penalty': 'l2', 'random_state': 42, 'solver': 'sag',
                'max_iter': 1000}
            self.model_params['ensembler'] = {'criterion': 'gini', 'splitter': 'best', 'max_features': 'sqrt',
                'random_state': 42, 'class_weight': 'balanced', 'ccp_alpha': 0.0}

            # Replace specified defaults and save the provided model parameters
            if parameters is not None:
                for model in parameters.keys():
                    for p in parameters[model].keys():
                        self.model_params[model][p] = parameters[model][p]

            # train the classifiers
            pred_df = self._fit_ensemble_model(train_data, 'DT', features, embedding_features)

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

        # Process target tasks into single categorization task
        data = self._target_processing(data)

        # Initialise prediction dataframe
        pred_df = deepcopy(data)

        # Predictions for baseline model
        if self.model_type == 'baseline':

            # Add predictions to the dataframe
            n_cols = len(pred_df.columns)
            for k in self.most_frequent_category:
                pred = self.most_frequent_category[k]
                pred_df.insert(loc=n_cols, column=f'{k}_prediction', value=pred)

        # Predictions for individual classifiers
        if self.model_type == 'random_forest' or self.model_type == 'svm' or self.model_type == 'logistic_regression':

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
                if self.model_type == 'random_forest':
                    y_pred = self.random_forest_classifier.predict(x_data)
                elif self.model_type == 'svm':
                    y_pred = self.multi_target_classifier.predict(x_data)
                else:
                    y_pred = self.logistic_regression_classifier.predict(x_data)
                y_pred = pd.DataFrame(y_pred, columns=self.task_cols)
                pred_df = deepcopy(data)
                n_cols = len(pred_df.columns)
                for t in self.task_cols:
                    pred_df.insert(loc=n_cols, column=f'{t}_prediction', value=y_pred[t].values)
                    n_cols += 1

            elif self.prediction_target == 'together':
                if self.model_type == 'random_forest':
                    y_pred = self.random_forest_classifier.predict(x_data)
                elif self.model_type == 'svm':
                    y_pred = self.multi_target_classifier.predict(x_data)
                else:
                    y_pred = self.logistic_regression_classifier.predict(x_data)
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

        # Predictions for ensembler models
        if self.model_type == 'ensembler_lr' or self.model_type == 'ensembler_dt':

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

            # Add submodel predictions to the ensembler input DF
            ensemble_input_df = deepcopy(data[['Target']])
            ensemble_features = []

            # SVM
            clf_svm = self.ensembler_models['SVM']
            y_pred_svm = clf_svm.predict_proba(x_data)
            # Add sub-model predictions to dataframe
            n = ensemble_input_df.shape[1]
            for i in range(y_pred_svm.shape[1]):
                col_title = f'SVM_{clf_svm.classes_[i]}'
                vals = y_pred_svm[:, i]
                ensemble_input_df.insert(loc=n, column=col_title, value=vals)
                ensemble_features.append(col_title)
                n += 1

            # Random Forest
            clf_rf = self.ensembler_models['random_forest_classifier']
            y_pred_rf = clf_rf.predict_proba(x_data)
            # Add sub-model predictions to dataframe
            n = ensemble_input_df.shape[1]
            for i in range(y_pred_rf.shape[1]):
                col_title = f'RF_{clf_rf.classes_[i]}'
                vals = y_pred_rf[:, i]
                ensemble_input_df.insert(loc=n, column=col_title, value=vals)
                ensemble_features.append(col_title)
                n += 1

            # Logistic Regression
            clf_lr = self.ensembler_models['logistic_regression_classifier']
            y_pred_lr = clf_lr.predict_proba(x_data)
            # Add sub-model predictions to dataframe
            n = ensemble_input_df.shape[1]
            for i in range(y_pred_lr.shape[1]):
                col_title = f'LR_{clf_lr.classes_[i]}'
                vals = y_pred_lr[:, i]
                ensemble_input_df.insert(loc=n, column=col_title, value=vals)
                ensemble_features.append(col_title)
                n += 1

            # TODO: add BERT classification model

            # Specify input for ensembler
            x_ensemble = ensemble_input_df[ensemble_features]

            # load the fit model
            clf = self.ensembler_models['ensembler']
            # Generate predictions on training data
            y_pred = clf.predict(x_ensemble)

            # Add ensemble predictions to sub-model
            n = ensemble_input_df.shape[1]
            ensemble_input_df.insert(loc=n, column='Ensemble_target', value=y_pred)

            # Save results
            self.submodel_results_test = deepcopy(ensemble_input_df)

            # Format predictions appropriately
            y_pred = pd.DataFrame(y_pred, index=x_ensemble.index, columns=['Target'])

            # Format predictions appropriately
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

    # # Imports
    # from src.data_processor import DataProcessor
    # from src.feature_engineering import FeatureEngineering
    #
    # # Load and clean the raw data
    # myDP = DataProcessor()
    # myDP.load_data(language='english', filepath='../data')  # May need to change to './data' or 'data' if on a Mac
    # myDP.clean_data()
    #
    # # Generate the features for model training
    # myFE = FeatureEngineering()
    # train_df = myFE.fit_transform(myDP.processed_data['train'], embedding_file_path='../data/glove.twitter.27B.25d.txt',
    #                               embedding_dim=25)
    # val_df = myFE.transform(myDP.processed_data['validation'])

    #TEMPORARY - REMOVE AFTER TESTING
    # Imports
    from src.feature_engineering import FeatureEngineering
    from src.classification_model import ClassificationModel
    import pickle as pkl
    import datetime
    # #
    # # Unpickle the pre-processed training data to load in future runs
    # train_data_file = 'data/processed_data/OLD-D4/dp_train.pkl'
    # with open(train_data_file, 'rb') as f:
    #     train_df = pkl.load(f)
    # #
    # # Unpickle the pre-processed validation data to load in future runs
    # val_data_file = 'data/processed_data/OLD-D4/dp_val.pkl'
    # with open(train_data_file, 'rb') as f:
    #     val_df = pkl.load(f)
    # #
    # # Generate the features for model training
    # myFE = FeatureEngineering()
    # # TODO: add more args - look at method definition
    # train_df = myFE.fit_transform(train_data=train_df,
    #                               embedding_file_path='data/glove.twitter.27B.25d.txt',
    #                               embedding_dim=25,
    #                               nrc_embedding_file='data/glove.twitter.27B.25d.txt',
    #                               slang_dict_path='data/SlangSD.txt',
    #                               stop_words_path='data/stopwords.txt',
    #                               language='english',
    #                               lexpath='/data/lexico_nrc.csv',
    #                               load_translations=True,
    #                               trans_path='data/translations.csv')
    # val_df = myFE.transform(val_df)
    # Unpickle the pre-processed training data to load in future runs
    train_data_file = '/Users/lindsayskinner/Documents/school/CLMS/573/data/D4/train_df.pkl'
    with open(train_data_file, 'rb') as f:
        train_df = pkl.load(f)
    #
    # Unpickle the pre-processed validation data to load in future runs
    val_data_file = '/Users/lindsayskinner/Documents/school/CLMS/573/data/D4/val_df.pkl'
    with open(train_data_file, 'rb') as f:
        val_df = pkl.load(f)

    # Instantiate the model
    myClassifier = ClassificationModel('logistic_regression')

    # Train the model
    features = ['percent_capitals', '!_count_normalized', '?_count_normalized', '$_count_normalized',
                '*_count_normalized', 'negative_ext', 'positive_ext', 'anger_ext', 'anticipation_ext', 'disgust_ext',
                'fear_ext', 'joy_ext', 'sadness_ext', 'surprise_ext', 'trust_ext', 'slangscore']
    embedding_features = ['Universal_Sentence_Encoder_embeddings', 'BERTweet_embeddings', 'Aggregate_embeddings']
    train_pred = myClassifier.fit(train_df,
                                  tasks=['hate_speech_detection', 'target_or_general', 'aggression_detection'],
                                  prediction_target='together',
                                  keep_training_data=False,
                                  features=features,
                                  embedding_features=embedding_features)

    # Run the model on the validation data
    val_pred = myClassifier.predict(val_df)

    # View a sample of the results
    train_df.head()
    val_df.head()


    import pickle as pkl
    file_path = '/Users/lindsayskinner/Documents/school/CLMS/573/data/D4/train_df.pkl'
    with open(file_path, 'wb') as f:
        pkl.dump(train_df, f)
    #
    file_path = '/Users/lindsayskinner/Documents/school/CLMS/573/data/D4/val_df.pkl'
    with open(file_path, 'wb') as f:
        pkl.dump(val_df, f)
