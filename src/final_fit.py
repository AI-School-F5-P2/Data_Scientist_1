import os
import pickle

from probability import false_positives_false_negatives
from testing import test_evaluation


def metrics_final_training(X, y):
    '''
    This function takes de whole dataset separated as X and y.
    It predicts y_predicted after passing X through the full_pipeline.
    It also prints the metrics obtained from the call of test_evaluation.
    '''
    path_to_full_pipeline = os.getenv('PATH_TO_FULL_PIPELINE')
    with open (path_to_full_pipeline, 'rb') as file:
        full_pipeline = pickle.load(file)
        
    y_predicted = full_pipeline.predict(X)

    probability = full_pipeline.predict_proba(X)
    
    #probabilities for positive class (1)
    y_probs = probability[:, full_pipeline.classes_.tolist().index(1)]

    test_evaluation(y, y_predicted, y_probs)

    false_positives_false_negatives(y, y_predicted, y_probs)