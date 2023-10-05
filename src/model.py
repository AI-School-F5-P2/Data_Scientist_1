from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import pickle
import os

def sampling_classifier_model(X_train_transformed, y_train):
    '''
    This function trains the model that includes the sampling technique.
    '''
    eec = EasyEnsembleClassifier(random_state=42)
    
    cv = StratifiedKFold(n_splits=5)
    
    eec_model_cv = cross_validate(eec, X_train_transformed, y_train, cv=cv, n_jobs=-1, scoring='recall')
    print(f"{eec_model_cv['test_score'].mean():.3f}")
    eec_model = EasyEnsembleClassifier(random_state=42).fit(X_train_transformed, y_train)
    
    return eec_model


def pickle_files(preprocessor, model):
    '''
    This function creates the pickle files: one for the preprocessor
    and another one for the final model.
    '''
    path_to_preprocessor = os.getenv('PATH_TO_PREPROCESSOR')
    with open(path_to_preprocessor, 'wb') as archivo:
        pickle.dump(preprocessor, archivo)
        print('Preprocessor file successfully created')

    path_to_model = os.getenv('PATH_TO_MODEL')
    with open(path_to_model, 'wb') as archivo:
        pickle.dump(model, archivo)
        print('Model file successfully created')