import numpy as np        # For numerical operations
import pandas as pd       # For data manipulation and analysis
import os
from sklearn.model_selection import train_test_split
# import sys
# import pathlib


from utils import get_logger, load_params

logger = get_logger(__name__)
# logger.debug(f"logger initialized with name: {__name__}")
params = load_params('params.yaml')
logger.debug("Methods successfuly loaded from utils file")

def load_data(data_url:str)->pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data Loaded from %s", data_url)
        return df
    except Exception as e:
        logger.error("Unexpected error occured while loading the data: %s", e)
        return
    
def data_processing(dataframe:pd.DataFrame)->pd.DataFrame:
    try:
        dataframe.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
        dataframe.rename(columns = {'v1':'target', 'v2':'text'}, inplace=True)
        logger.debug('Data Processing completed')
        return dataframe
    except Exception as e:
        logger.error(f'Unexpected Error occured while data processing {e}')
        raise ValueError (f'Unexpected Error occured while data processing {e}')
    
def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str)->None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train_data.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test_data.csv'), index=False)
        logger.debug('Train and Test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error while saving the data %s', e)

def main():
    try:
        # test_size = 0.2
        test_size = params['data_ingestion']['test_size']
        data_url='https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/refs/heads/main/experiments/spam.csv'
        df = load_data(data_url=data_url)
        processed_data = data_processing(df)
        train_data, test_data = train_test_split(processed_data, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process %s', e)

if __name__ =="__main__":
    main()