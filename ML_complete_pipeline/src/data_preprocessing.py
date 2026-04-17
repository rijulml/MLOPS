import pandas as pd
import logging
import os
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import string


# Logging 
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')

logger.setLevel('DEBUG')

filepath = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(filepath)
file_handler.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)


def preprocess_data(df:pd.DataFrame, text_column:str, target_column:str)->pd.DataFrame:
    try:

        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates Removed')

        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')

        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column Encoded')
        return df
    except Exception as e:
        logger.error("Unexpected error occured while processing data %s", e)

def main():
    try:
        train_data = pd.read_csv('./data/raw/train_data.csv')
        test_data = pd.read_csv('./data/raw/test_data.csv')
        logger.debug('Data loaded Successfuly')

        processed_train_data = preprocess_data(train_data, 'text', 'target')
        processed_test_data = preprocess_data(test_data,'text', 'target')

        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        processed_train_data.to_csv(os.path.join(data_path, 'train_processed_data.csv'), index=False)
        processed_test_data.to_csv(os.path.join(data_path, 'test_processed_data.csv'), index=False)

        logger.debug('Processed data save at path %s', data_path)
    except Exception as e:
        logger.error('Unexpected error occured while processing data: %s', e)


if __name__=="__main__":
    main()