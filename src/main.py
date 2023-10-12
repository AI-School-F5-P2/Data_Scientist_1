from utils import load_data, X_y_separation, split_stratify_y, create_test_set
from preprocessing import create_fit_preprocessor
from model import sampling_classifier_model, pickle_files
from testing import transform_predict_test_data
from decouple import config

if __name__ == '__main__':
    
    path_to_data = config('PATH_TO_DATA_CSV')
    
    df = load_data(path_to_data)

    X, y = X_y_separation(df, 'stroke')

    X_train, X_test, y_train, y_test = split_stratify_y(X, y, test_size=0.2)

    create_test_set(X_test, y_test)

    preprocessor, X_train_transformed = create_fit_preprocessor(X_train)

    model = sampling_classifier_model(X_train_transformed, y_train)

    pickle_files(preprocessor, model)
    
    transform_predict_test_data()