import pandas as pd
from utils import load_data, X_y_separation, split_stratify_y, create_test_set
from preprocessing import create_preprocessor
from model import sampling_classifier_model, pickle_files
from testing import transform_test_data, predict_test_data, test_evaluation
from decouple import config

if __name__ == '__main__':
    
    path_to_data = config('PATH_TO_DATA_CSV')
    
    df = load_data(path_to_data)

    X, y = X_y_separation(df, 'stroke')

    X_train, X_test, y_train, y_test = split_stratify_y(X, y, test_size=0.2)

    create_test_set(X_test, y_test)

    preprocessor = create_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)

    model = sampling_classifier_model(X_train_transformed, y_train)

    pickle_files(preprocessor, model)

    X_test_transformed, y_test_true = transform_test_data()

    y_test_predicted, y_probs = predict_test_data(X_test_transformed)

    test_evaluation(y_test_true, y_test_predicted, y_probs)