import json
import os
import re


from data_processor import DataProcessor
from feature_engineering import FeatureEngineering
from classification_model import ClassificationModel


def load_config(config_path):
    """
    Load config from a given path.
    
    Args:
        config_path (str): The path to the config file.

    Returns:
        dict: The loaded config if successful, otherwise None.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def main(config):
    doc_config = config['document_processing']
    input_tsv_files = doc_config['input_tsv_files']
  
    # Initialize the class
    myDP = DataProcessor()
    # Load data from disk
    myDP.load_data(language= input_tsv_files['language'], train_file = input_tsv_files['training'], validation_file = input_tsv_files['devtest'])  
    # Clean the text
    myDP.clean_data()
  
    # View a sample of the results
    # myDP.processed_data['train'].head()
    # myDP.processed_data['validation'].head()
    myFE = FeatureEngineering()
      


if __name__ == "__main__":
    config = load_config(config_path= os.path.join('..', 'config.json'))
    main(config)
