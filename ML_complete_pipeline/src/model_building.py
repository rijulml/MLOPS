import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import logging
import numpy as np
import pickle

# Logging 
log_dir = '../logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_training')

logger.setLevel('DEBUG')

filepath = os.path.join(log_dir,'model_training.log')
file_handler = logging.FileHandler(filepath)
file_handler.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and Nan filled for the file loaded from path %s', file_path)
        return df
    except Exception as e:
        logger.error("Unexpected Error while loading or filling Nan: %s", e)

def train_model(X_train:np.ndarray, y_train:np.ndarray, n_estimators:int, random_state:int) -> RandomForestClassifier:

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug("Initializing Random Forest model with paramerters")

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        logger.debug("Model training started with %d samples", X_train.shape[0])

        clf.fit(X_train, y_train)
        logger.debug("Model Training completed")
        return clf
    except Exception as e:
        logger.error("Unexpected error occured while training: %s", e)

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # params = load_params('params.yaml')['model_building']
        n_estimators=100
        random_state=42
        train_data = load_data('../data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, n_estimators, random_state)
        
        model_save_path = '../models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()