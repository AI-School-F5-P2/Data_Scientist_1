import numpy as np
from sklearn.metrics import confusion_matrix


def false_positives_false_negatives(y_true, y_predicted, y_probs):
    '''
    This function takes y true, y predicted by the model and y_probs 
    which contains the probabilities of the postive class (1). It calculates
    the probabilities of the false positives and false negatives to determine
    the range where the model does not predict with certainty.
    '''
    conf_matrix = confusion_matrix(y_true, y_predicted)
    print('')
    print('Confusion Matrix:')
    print(conf_matrix)

    false_positives_idx = np.where((y_true == 0) & (y_predicted == 1))
    false_negatives_idx = np.where((y_true == 1) & (y_predicted == 0))

    probs_fp = y_probs[false_positives_idx]
    probs_fn = y_probs[false_negatives_idx]

    print('')
    print("False Positive Probabilities:")
    print(probs_fp)

    print('')
    print("False Negative Probabilities:")
    print(probs_fn)

    all_probs = np.concatenate((probs_fp, probs_fn), axis = 0)

    min_prob = np.min(all_probs)
    max_prob = np.max(all_probs)

    print('')
    print(f'Min Probability: {min_prob:.3f}')
    print(f'Max Probability: {max_prob:.3f}')