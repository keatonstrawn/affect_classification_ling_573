""" This script defines a helper class to load and clean the HatEval data for task 5 of the 2019 SemEval shared tasks.
"""

# Libraries
import re
import emoji
import pandas as pd

from typing import Optional, Dict, Tuple, List
from copy import deepcopy


# Define class to handle data processing
class DataProcessor:
    def __init__(self):
        """Processes the raw HateEval data for use in hate speech detection tasks A and B, as specified in SemEval 2019
        task 5.

        Includes methods for the following data preparation tasks:
            * Pulling the data from GitHub
            * Loading the data from disk
            * Cleaning the text (removing hyperlinks, emojis, etc.)

            Attributes
            ----------
            raw_data
                The raw, unprocessed data. Stored in a dictionary that is keyed for 'train', 'validation' and 'test'
                datasets.
            processed_data
                The processed data, in a format that is compatible with the FeatureEngineering input. Stored in a
                dictionary that is keyed for 'train', 'validation' and 'test' datasets.
        """
        self.raw_data: Dict[str: pd.DataFrame] = None
        self.processed_data: Dict[str: pd.DataFrame] = None

    def load_data(self, load_from_disk: bool = False, language: str = 'english', filepath: Optional[str] = None,
                  train_file: Optional[str] = None, validation_file: Optional[str] = None,
                  test_file: Optional[str] = None) -> None:
        """Loads the raw data into the data processor.

        Arguments:
        ----------
            load_from_disk
                Indicates whether the data is to be loaded from disk (True) or downloaded from GitHub
            filepath
                If data is to be loaded from disk, the filepath to the directory containing the data.
            train_file
                The name of the file containing the training data. If no name is provided then this is assumed to be
                train_en.tsv.
            validation_file
                The name of the file containing the validation data. If no name is provided then this is assumed to be
                dev_en.tsv.
            test_file
                The name of the file containing the test data. If no name is provided then this is assumed to be
                test_en.tsv.
            language
                The language data to be downloaded. Can be 'english' or 'spanish' or 'both'.
        """

        if load_from_disk:

            # Ensure filepath is specified
            assert filepath is not None, 'Must specify filepath in order to load the data from disk.'
            assert language == 'english' or language == 'spanish', "Language must be one of 'english' or 'spanish'"

            # Fill in filenames, if missing
            if language == 'english':
                if train_file is None: train_file = 'train_en.tsv'
                if validation_file is None: validation_file = 'dev_en.tsv'
                if test_file is None: test_file = 'test_en.tsv'
            if language == 'spanish':
                if train_file is None: train_file = 'train_es.tsv'
                if validation_file is None: validation_file = 'dev_es.tsv'
                if test_file is None: test_file = 'test_es.tsv'

            # Load data from disk
            raw_train_df = pd.read_csv(f'{filepath}/{train_file}', sep='\t', header=0, index_col='id')
            raw_val_df = pd.read_csv(f'{filepath}/{validation_file}', sep='\t', header=0, index_col='id')
            raw_test_df = pd.read_csv(f'{filepath}/{test_file}', sep='\t', header=0, index_col='id')

            # Add column specifying language:
            if language == 'english':
                lang_id = 'en'
            elif language == 'spanish':
                lang_id = 'es'
            raw_train_df = raw_train_df.assign(language=lang_id)
            raw_val_df = raw_val_df.assign(language=lang_id)
            raw_test_df = raw_test_df.assign(language=lang_id)

        else:

            # Load English data only
            if language == 'english':
                train_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-English-B/train_en.tsv'
                val_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-English-B/dev_en.tsv'
                test_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%233%20Evaluation-English-A/test_en.tsv'

                raw_train_df = pd.read_csv(train_url, sep='\t', header=0, index_col='id')
                raw_val_df = pd.read_csv(val_url, sep='\t', header=0, index_col='id')
                raw_test_df = pd.read_csv(test_url, sep='\t', header=0, index_col='id')

                # Add column specifying language:
                raw_train_df = raw_train_df.assign(language='en')
                raw_val_df = raw_val_df.assign(language='en')
                raw_test_df = raw_test_df.assign(language='en')

            # Load Spanish data only
            elif language == 'spanish':
                train_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-Spanish-B/train_es.tsv'
                val_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-Spanish-B/dev_es.tsv'
                test_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%233%20Evaluation-Spanish/test_es.tsv'

                raw_train_df = pd.read_csv(train_url, sep='\t', header=0, index_col='id')
                raw_val_df = pd.read_csv(val_url, sep='\t', header=0, index_col='id')
                raw_test_df = pd.read_csv(test_url, sep='\t', header=0, index_col='id')

                # Add column specifying language:
                raw_train_df = raw_train_df.assign(language='es')
                raw_val_df = raw_val_df.assign(language='es')
                raw_test_df = raw_test_df.assign(language='es')

            # Load both English and Spanish data and concatenate the dataframes
            else:
                # Load English data
                train_en_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-English-B/train_en.tsv'
                val_en_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-English-B/dev_en.tsv'
                test_en_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%233%20Evaluation-English-A/test_en.tsv'

                en_train_df = pd.read_csv(train_en_url, sep='\t', header=0, index_col='id')
                en_val_df = pd.read_csv(val_en_url, sep='\t', header=0, index_col='id')
                en_test_df = pd.read_csv(test_en_url, sep='\t', header=0, index_col='id')

                # Add column specifying language:
                en_train_df = en_train_df.assign(language='en')
                en_val_df = en_val_df.assign(language='en')
                en_test_df = en_test_df.assign(language='en')

                # Load Spanish data
                train_es_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-Spanish-B/train_es.tsv'
                val_es_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-Spanish-B/dev_es.tsv'
                test_es_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%233%20Evaluation-Spanish/test_es.tsv'

                es_train_df = pd.read_csv(train_es_url, sep='\t', header=0, index_col='id')
                es_val_df = pd.read_csv(val_es_url, sep='\t', header=0, index_col='id')
                es_test_df = pd.read_csv(test_es_url, sep='\t', header=0, index_col='id')

                # Add column specifying language:
                es_train_df = es_train_df.assign(language='es')
                es_val_df = es_val_df.assign(language='es')
                es_test_df = es_test_df.assign(language='es')

                # Concatenate the dataframes
                raw_train_df = pd.concat([en_train_df, es_train_df])
                raw_val_df = pd.concat([en_val_df, es_val_df])
                raw_test_df = pd.concat([en_test_df, es_test_df])

        # Save raw data
        self.raw_data = {'train': raw_train_df, 'validation': raw_val_df, 'test': raw_test_df}

    def _remove_urls(self, tweet: str) -> str:
        """Removes urls from the text of a particular tweet.

        Arguments:
        ----------
        tweet
            The tweet text.

        Returns:
        --------
        tweet text with all urls removed
        """
        tweet_list = tweet.split()
        url_list = [w for w in tweet_list if w.startswith('http')]
        cleaned_tweet = [w for w in tweet_list if w not in url_list]
        cleaned_tweet = ' '.join(cleaned_tweet)

        return cleaned_tweet

    def _separate_hashtags(self, tweet: str) -> Tuple[str, List[str]]:
        """Separates hashtags from the tweet text and creates a list of all included hashtags.

        Arguments:
        ----------
        tweet
            The tweet text.

        Returns:
        --------
        a tuple of the tweet text, with all hashtags removed, and a separate list of the affiliated hashtags
        """

        tweet_list = tweet.split()
        hashtag_list = [w for w in tweet_list if w.startswith('#')]
        cleaned_tweet = [w for w in tweet_list if w not in hashtag_list]
        cleaned_tweet = ' '.join(cleaned_tweet)

        return cleaned_tweet, hashtag_list

    def _separate_usernames(self, tweet:str) -> Tuple[str, List[str]]:
        """Separates twitter usernames from the tweet text and creates a list of all referenced usernames.

        Arguments:
        ----------
        tweet
            The tweet text.

        Returns:
        --------
        a tuple of the tweet text, with usernames replaced by 'user' , and a separate list of the affiliated usernames
        """

        tweet_list = tweet.split()
        user_list = [w for w in tweet_list if w.startswith('@')]
        cleaned_tweet = [w if w not in user_list else 'user' for w in tweet_list]
        cleaned_tweet = ' '.join(cleaned_tweet)

        return cleaned_tweet, user_list

    def _replace_emojis(self, tweet: str, language: str = 'en') -> str:
        """Replaces any emojis that appear in the text with their (English) name.

        Arguments:
        ----------
        tweet
            The tweet text.
        language
            The primary language that the tweet is written in. 'en' specified English and 'es' specifies Spanish.

        Returns:
        --------
        the tweet text with all emojis replaced with their names
        """

        demojized_tweet = emoji.demojize(tweet, language=language)
        tweet_list = demojized_tweet.split(':')
        cleaned_tweet_list = []
        for w in tweet_list:
            w = w.replace('_', ' ')
            cleaned_tweet_list.append(w)
        cleaned_tweet = ' '.join(cleaned_tweet_list)

        return cleaned_tweet

    def _remove_and_count_punctuation(self, tweet: str, symbol_list: Optional[List[str]] = None) -> Tuple[str, dict]:
        """Removes all punctuation from the tweet text and returns the count of the specified punctuation symbols.

        Arguments:
        ----------
        tweet
            The tweet text.
        symbol_list
            The list of punctuation symbols for which to keep count.

        Returns:
        --------
        a tuple of the tweet text, with punctuation removed, and a separate dictionary of the counts for the symbols
        specified in symbol_list.
        """

        # Fill in default symbols, if missing
        if symbol_list is None:
            symbol_list = ('!', '?', '$', '*')

        cleaned_tweet = ''
        symbol_counts = {f'{s}_count': [0] for s in symbol_list}
        for char in tweet:
            if char in symbol_list:
                symbol_counts[f'{char}_count'][0] += 1
            else:
                cleaned_tweet += char

        return cleaned_tweet, symbol_counts

    def _get_capital_perc_and_lowercase(self, tweet: str) -> Tuple[str, float]:
        """Lowercases the tweet text and returns the percentage of the alphabet characters in the tweet that were
        capitalized.

        Arguments:
        ----------
        tweet
            The tweet text.

        Returns:
        --------
        a tuple of the lower-cased tweet text and a float indicate what percentage of alphabet characters in the
        tweet were capitalized.
        """

        capital_ct = len(re.findall(r'[A-Z]', tweet))
        alpha_ct = len(re.findall(r'[A-Za-z]', tweet))
        capital_pct = capital_ct / alpha_ct

        cleaned_tweet = tweet.lower()

        return cleaned_tweet, capital_pct

    def clean_tweet(self, tweet: str, language: str, symbol_list: Optional[List[str]] = None) -> pd.DataFrame:
        """Cleans the text of a single tweet and returns all accumulated information and cleaned text in a dataframe.

        Arguments:
        ---------
        tweet
            The tweet text.
        language
            The primary language that the tweet is written in. 'en' specified English and 'es' specifies Spanish.
        symbol_list
            The list of punctuation symbols for which to keep count.

        Returns:
        -------
        A dataframe containing the cleaned tweet text, hashtag list, username list, punctuation count and percent of
        text that was capitalized.
        """

        # Remove URls
        cleaned_tweet = self._remove_urls(tweet)

        # Separate hashtags
        cleaned_tweet, hashtag_list = self._separate_hashtags(cleaned_tweet)

        # Separate Twitter user IDs
        cleaned_tweet, user_list = self._separate_usernames(cleaned_tweet)

        # Get punctuation count and remove punctuation
        cleaned_tweet, symbol_cts = self._remove_and_count_punctuation(cleaned_tweet, symbol_list)

        # Get percent of the tweet that is capitalized and lowercase the tweet
        cleaned_tweet, capital_pct = self._get_capital_perc_and_lowercase(cleaned_tweet)

        # Replace emojis with names
        cleaned_tweet = self._replace_emojis(cleaned_tweet, language)
        # need to lowercase again, as some emoji names are capitalized
        cleaned_tweet = cleaned_tweet.lower()

        # Construct dataframe of results
        res_df = {'cleaned_text': [cleaned_tweet], 'hashtags': [hashtag_list], 'user_ids': [user_list],
                  'percent_capitals': [capital_pct]}
        res_df.update(symbol_cts)
        res_df = pd.DataFrame(res_df)

        return res_df

    def clean_data(self, symbol_list: Optional[List[str]] = None) -> None:
        """Cleans the raw text data and saves the cleaned data as a new column on the cleaned_data dataframe. Also
        stores any additional data accumulated during cleaning as new columns on the cleaned_data dataframe.

        Arguments:
        ---------
        symbol_list
            The list of punctuation symbols for which to keep count.
        """

        # Generate a copy of the raw data to be processed
        raw_train = deepcopy(self.raw_data['train'])
        raw_val = deepcopy(self.raw_data['validation'])
        raw_test = deepcopy(self.raw_data['test'])

        # Get dataframe of new, processed columns
        processed_train_data = pd.DataFrame()
        processed_val_data = pd.DataFrame()
        processed_test_data = pd.DataFrame()

        # Process English data
        if 'en' in raw_train['langauge']:

            # training data
            en_raw_train = raw_train[raw_train['language'] == 'en']
            en_processed_train = en_raw_train['text'].apply(self.clean_tweet, language='en')
            en_processed_train = pd.concat([en_raw_train, en_processed_train], axis=1)
            processed_train_data = pd.concat([processed_train_data, en_processed_train], axis=0)

            # validation data
            en_raw_val = raw_val[raw_val['language'] == 'en']
            en_processed_val = en_raw_val['text'].apply(self.clean_tweet, language='en')
            en_processed_val = pd.concat([en_raw_val, en_processed_val], axis=1)
            processed_val_data = pd.concat([processed_val_data, en_processed_val], axis=0)

            # test data
            en_raw_test = raw_train[raw_test['language'] == 'en']
            en_processed_test = en_raw_test['text'].apply(self.clean_tweet, language='en')
            en_processed_test = pd.concat([en_raw_test, en_processed_test], axis=1)
            processed_test_data = pd.concat([processed_test_data, en_processed_test], axis=0)

        # Process Spanish data
        if 'es' in raw_train['langauge']:

            # training data
            es_raw_train = raw_train[raw_train['language'] == 'es']
            es_processed_train = es_raw_train['text'].apply(self.clean_tweet, language='es')
            es_processed_train = pd.concat([es_raw_train, es_processed_train], axis=1)
            processed_train_data = pd.concat([processed_train_data, es_processed_train], axis=0)

            # validation data
            es_raw_val = raw_val[raw_val['language'] == 'es']
            es_processed_val = es_raw_val['text'].apply(self.clean_tweet, language='es')
            es_processed_val = pd.concat([es_raw_val, es_processed_val], axis=1)
            processed_val_data = pd.concat([processed_val_data, es_processed_val], axis=0)

            # test data
            es_raw_test = raw_train[raw_test['language'] == 'es']
            es_processed_test = es_raw_test['text'].apply(self.clean_tweet, language='es')
            es_processed_test = pd.concat([es_raw_test, es_processed_test], axis=1)
            processed_test_data = pd.concat([processed_test_data, es_processed_test], axis=0)

        # Rename text field so that it's clear it contains the raw, unprocessed text
        processed_train_data.rename({'text': 'raw_text'}, inplace=True)
        processed_val_data.rename({'text': 'raw_text'}, inplace=True)
        processed_test_data.rename({'text': 'raw_text'}, inplace=True)

        self.processed_data = {'train': processed_train_data, 'validation': processed_val_data,
                               'test': processed_test_data}


    # TODO: Is there any way to map slang terms and masked-swear words (e.g. f***) to actual word that doesn't involve
    #  compiling our own dictionary of terms? And/or is there a thesaurus of slang terms so we can use just one form?


# Example of how to use the DataProcessor class
if __name__ == '__main__':

    # Initialize the class
    myDP = DataProcessor()

    # Load data from disk
    myDP.load_data(load_from_disk=True, language='english', filepath='/Users/lindsayskinner/Documents/school/CLMS/573/data')

    # Clean the text
    myDP.clean_data()

    # View the results
    myDP.processed_data['train'].head()


    # ### TEMPORARY - REMOVE ONCE TESTING FINISHED ###
    #
    tmp = myDP.raw_data['train'].loc[201, 'text']
    tmp2 = myDP.raw_data['train'].loc[304, 'text']

    myDP.clean_tweet(tmp, 'en')
    myDP.clean_tweet(tmp2, 'en')
    #
    # # Remove URls
    # cleaned_tweet = myDP._remove_urls(tmp)
    #
    # # Separate hashtags
    # cleaned_tweet, hashtag_list = myDP._separate_hashtags(cleaned_tweet)
    #
    # # Separate Twitter user IDs
    # cleaned_tweet, user_list = myDP._separate_usernames(cleaned_tweet)
    #
    # # Get punctuation count and remove punctuation
    # cleaned_tweet, symbol_cts = myDP._remove_and_count_punctuation(cleaned_tweet)
    #
    # # Get percent of the tweet that is capitalized and lowercase the tweet
    # cleaned_tweet, capital_pct = myDP._get_capital_perc_and_lowercase(cleaned_tweet)
    #
    # # Replace emojis with names
    # cleaned_tweet = myDP._replace_emojis(cleaned_tweet, language='en')
    # # need to lowercase again, as some emoji names are capitalized
    # cleaned_tweet = cleaned_tweet.lower()
    #
    # # Construct dataframe of results
    # res_df = {'cleaned_text': [cleaned_tweet], 'hashtags': [hashtag_list], 'user_ids': [user_list],
    #           'percent_capitals': [capital_pct]}
    # res_df.update(symbol_cts)
    # res_df2 = pd.DataFrame(res_df)
