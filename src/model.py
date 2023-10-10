import numpy as np
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
import pickle
import os


def sampling_classifier_model(X_train_transformed, y_train):
    '''
    This function trains the model that includes the sampling technique.
    '''
    eec = EasyEnsembleClassifier(random_state=42)
    
    cv = StratifiedKFold(n_splits=5)

    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    eec_model_cv = cross_validate(eec, X_train_transformed, y_train, cv=cv, scoring=scoring)

    print('')
    print('Cross Validation Metrics:')
    print(f"Accuracy: {(np.mean(eec_model_cv['test_accuracy'])):.3f}")
    print(f"Precision: {(np.mean(eec_model_cv['test_precision'])):.3f}")
    print(f"Recall: {(np.mean(eec_model_cv['test_recall'])):.3f}")
    print(f"F1-Score: {(np.mean(eec_model_cv['test_f1'])):.3f}")
    print('')
 
    eec_model = EasyEnsembleClassifier(random_state=42).fit(X_train_transformed, y_train)
    
    return eec_model


def pickle_files(preprocessor, model):
    '''
    This function creates the full_pipeline pickle file. It takes 
    the preprocessor and the model previously trained and creates 
    the full pipeline (preprocessing + estimator).
    '''   
    full_pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    path_to_full_pipeline = os.getenv('PATH_TO_FULL_PIPELINE')
    with open(path_to_full_pipeline, 'wb') as file:
        pickle.dump(full_pipeline, file)
        print('Full pipeline file successfully created')
