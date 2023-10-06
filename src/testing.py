import pandas as pd
from utils import X_y_separation
import os
import pickle

def transform_test_data():
    '''
    This function reads the test.csv file, transforms the X variables
    using the preprocessor pickle file. It returns X_test_transformed.
    '''
    path_to_open = os.getenv('PATH_TO_SAVE_CSV')
    df_test = pd.read_csv(path_to_open)

    #drop the column Unnamed: 0 that was created automatically
    #when saving the file the first time.
    df_test = df_test.drop(columns=['Unnamed: 0'])
    X_test, y_test = X_y_separation(df_test, 'stroke')

    path_to_preprocessor = os.getenv('PATH_TO_PREPROCESSOR')
    with open (path_to_preprocessor, 'rb') as archivo:
        preprocessor = pickle.load(archivo)
    
    X_test_transformed = preprocessor.transform(X_test)

    return X_test_transformed, y_test


def predict_test_data(X_test_transformed):
    '''
    This function takes as argument the X_test_transformed and returns
    the y_test_predicted by the model using the pickle file.
    '''
    path_to_model = os.getenv('PATH_TO_MODEL')
    with open(path_to_model, 'rb') as archivo:
        model = pickle.load(archivo)
    
    y_test_predicted = model.predict(X_test_transformed)
    return y_test_predicted


def test_evaluation(y_test_true, y_test_predicted):
    pass




    