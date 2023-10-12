from utils import load_data, X_y_separation, split_stratify_y, create_test_set
from preprocessing import create_fit_preprocessor
from model import sampling_classifier_model, pickle_files
from testing import transform_predict_test_data
from final_fit import metrics_final_training
import app

from dotenv import load_dotenv
import os


if __name__ == '__main__':
    
    '''
    TRAIN - TEST SEPARATION FOR EXPERIMENTATION
    '''
    #loading of .env variables and data
    load_dotenv()
    path_to_data = os.getenv('PATH_TO_DATA_CSV')
    df = load_data(path_to_data)

    X, y = X_y_separation(df, 'stroke')

    X_train, X_test, y_train, y_test = split_stratify_y(X, y, test_size=0.2)

    create_test_set(X_test, y_test)

    preprocessor, X_train_transformed = create_fit_preprocessor(X_train)

    model = sampling_classifier_model(X_train_transformed, y_train)

    pickle_files(preprocessor, model)
    
    transform_predict_test_data()


    '''
    WHOLE DATASET FOR FINAL MODEL TRAINING
    '''
    print('-'*100)
    #loading of .env variables and data
    load_dotenv()
    path_to_data = os.getenv('PATH_TO_DATA_CSV')
    df = load_data(path_to_data)

    X, y = X_y_separation(df, 'stroke')

    preprocessor, X_transformed = create_fit_preprocessor(X)

    model = sampling_classifier_model(X_transformed, y)

    pickle_files(preprocessor, model)

    metrics_final_training(X, y)
    