""" This script defines a helper class to load and clean the HatEval data for task 5 of the 2019 SemEval shared tasks.
"""

# Libraries
import re
import emoji
import pandas as pd

from typing import Optional, Dict, Tuple, List


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

    def load_data(self, load_from_disk: bool = False, filepath: Optional[str] = None, train_file: Optional[str] = None,
                  validation_file: Optional[str] = None, test_file: Optional[str] = None,
                  language: Optional[str] = 'English') -> None:
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
                The language data to be downloaded, if not loading data from disk. Can be 'english', 'spanish', or
                'all'.
        """

        if load_from_disk:

            # Ensure filepath is specified
            assert filepath is not None, 'Must specify filepath in order to load the data from disk.'

            # Fill in filenames, if missing
            if train_file is None: train_file = 'train_en.tsv'
            if validation_file is None: validation_file = 'dev_en.tsv'
            if test_file is None: test_file = 'test_en.tsv'

            # Load data from disk
            raw_train_df = pd.read_csv(f'{filepath}/{train_file}', sep='\t', header=0)
            raw_val_df = pd.read_csv(f'{filepath}/{validation_file}', sep='\t', header=0)
            raw_test_df = pd.read_csv(f'{filepath}/{test_file}', sep='\t', header=0)

        else:

            # Load English data only
            if language == 'english':
                train_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-English-B/train_en.tsv'
                val_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-English-B/dev_en.tsv'
                test_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%233%20Evaluation-English-A/test_en.tsv'

                raw_train_df = pd.read_csv(train_url, sep='\t', header=0)
                raw_val_df = pd.read_csv(val_url, sep='\t', header=0)
                raw_test_df = pd.read_csv(test_url, sep='\t', header=0)

            # Load Spanish data only
            elif language == 'spanish':
                train_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-Spanish-B/train_es.tsv'
                val_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-Spanish-B/dev_es.tsv'
                test_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%233%20Evaluation-Spanish/test_es.tsv'

                raw_train_df = pd.read_csv(train_url, sep='\t', header=0)
                raw_val_df = pd.read_csv(val_url, sep='\t', header=0)
                raw_test_df = pd.read_csv(test_url, sep='\t', header=0)

            # Load both English and Spanish data and concatenate the dataframes
            else:
                # Load English data
                train_en_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-English-B/train_en.tsv'
                val_en_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-English-B/dev_en.tsv'
                test_en_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%233%20Evaluation-English-A/test_en.tsv'

                en_train_df = pd.read_csv(train_en_url, sep='\t', header=0)
                en_val_df = pd.read_csv(val_en_url, sep='\t', header=0)
                en_test_df = pd.read_csv(test_en_url, sep='\t', header=0)

                # Load Spanish data
                train_es_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-Spanish-B/train_es.tsv'
                val_es_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%232%20Development-Spanish-B/dev_es.tsv'
                test_es_url = 'https://github.com/cicl2018/HateEvalTeam/blob/master/Data%20Files/Data%20Files/%233%20Evaluation-Spanish/test_es.tsv'

                es_train_df = pd.read_csv(train_es_url, sep='\t', header=0)
                es_val_df = pd.read_csv(val_es_url, sep='\t', header=0)
                es_test_df = pd.read_csv(test_es_url, sep='\t', header=0)

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

    def _replace_emojis(self, tweet: str, language: str = 'English') -> str:
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

    def _remove_and_count_punctuation(self, tweet: str, symbol_list: List[str] = ('!', '?', '$', '*')
                                      ) -> Tuple[str, dict]:
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

        cleaned_tweet = ''
        symbol_counts = {s: 0 for s in symbol_list}
        for char in tweet:
            if char in symbol_list:
                symbol_counts[char] += 1
            else:
                cleaned_tweet += char

        return cleaned_tweet, symbol_counts

    def _get_capitals_perc_and_lowercase(self, tweet: str) -> Tuple[str, float]:
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



    # TODO: Is there any way to map slang terms and masked-swear words (e.g. f***) to actual word that doesn't involve
    #  compiling our own dictionary of terms?




# Example of how to use the DataProcessor class
if __name__ == '__main__':
    2+2