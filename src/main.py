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
    myDP.load_data(language= input_tsv_files['language'],
                   filepath = input_tsv_files['filepath']) 
                #    train_file = input_tsv_files['training'], 
                #    validation_file = input_tsv_files['devtest'])  
                
    # Clean the text
    myDP.clean_data()
  
    # Instantiate the FeatureEngineering object
    myFE = FeatureEngineering(config['model']['feature_engineering']['approach'])
    # Fit
    train_df = myFE.fit_transform(myDP.processed_data['train'])
    # Transform
    val_df = myFE.transform(myDP.processed_data['validation'])

    # View a sample of the results
    # with open("test.txt", "w") as f:
    #     f.write(str(train_df.head()))
    #     f.write("\t")
    #     f.write(str(val_df.head()))

    # Instantiate the model
    myClassifier = ClassificationModel(config['model']['classification']['approach'])
    
    # Train the model
    train_pred = myClassifier.fit(train_df, tasks=[config['model']['classification']['params']['task1']], 
                                  keep_training_data=config['model']['classification']['params']['keep_training_data'])

    # Run the model on the validation data
    val_pred = myClassifier.predict(val_df)

    # View a sample of the results
    train_df.head()
    val_df.head()      


if __name__ == "__main__":
    config = load_config(config_path= os.path.join('..', 'config.json'))
    main(config)
