import pandas as pd
import numpy as np
import logging
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Union
from utils import get_logger, load_params

logger = get_logger(__name__)
params = load_params('params.yaml')

def load_data(file_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and Nan filled for the file loaded from path %s', file_path)
        return df
    except Exception as e:
        logger.error("Unexpected Error while loading or filling Nan: %s", e)

def apply_tfidf(text_col:str, max_features:int, df:pd.DataFrame,target_col:str)->pd.DataFrame:
    try:
        tfid = TfidfVectorizer(max_features = max_features)
        X = tfid.fit_transform(df[text_col]).toarray()
        transformed_df = pd.DataFrame(X)
        transformed_df['label'] = df[target_col].values
        return transformed_df
    except Exception as e:
        pass

def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str)->None:
    try:
        raw_data_path = os.path.join(data_path, 'processed')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train_tfidf.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test_tfidf.csv'), index=False)
        logger.debug('Train and Test processed data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error while saving the data %s', e)

def main():
    try:
        max_features = params['feature_engineering']['max_features']
        train_data = load_data('./data/interim/train_processed_data.csv')
        test_data = load_data('./data/interim/test_processed_data.csv')

        train_tfidf = apply_tfidf('text', max_features, train_data, 'target')
        test_tfidf = apply_tfidf('text', max_features, test_data, 'target')

        save_data(train_tfidf, test_tfidf, './data')
    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)


if __name__ =="__main__":
    main()
