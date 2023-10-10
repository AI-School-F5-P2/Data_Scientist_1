import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score
from utils import X_y_separation

import os
import pickle

from probability import false_positives_false_negatives


def transform_predict_test_data():
    '''
    This function transforms the test.csv file and predicts
    the response (y_test_predicted). It calls the evaluation 
    function to obtain performance metrics.
    '''
    path_to_open = os.getenv('PATH_TO_SAVE_CSV')
    df_test = pd.read_csv(path_to_open)

    #drop the column Unnamed: 0 that was created automatically
    #when saving the file the first time.
    df_test = df_test.drop(columns=['Unnamed: 0'])
    X_test, y_test = X_y_separation(df_test, 'stroke')

    path_to_full_pipeline = os.getenv('PATH_TO_FULL_PIPELINE')
    with open (path_to_full_pipeline, 'rb') as file:
        full_pipeline = pickle.load(file)
    
    y_test_predicted = full_pipeline.predict(X_test)

    probability = full_pipeline.predict_proba(X_test)

    #probabilities for positive class (1)
    y_probs = probability[:, full_pipeline.classes_.tolist().index(1)]

    test_evaluation(y_test, y_test_predicted, y_probs)


def test_evaluation(y_test_true, y_test_predicted, y_probs):
    '''
    This function takes the y that is true and the y predicted by
    the model and calculates the different metrics.
    '''
    acc = accuracy_score(y_test_true, y_test_predicted)
    conf_matrix = confusion_matrix(y_test_true, y_test_predicted)
    precision = precision_score(y_test_true, y_test_predicted)
    recall = recall_score(y_test_true, y_test_predicted)
    f1 = f1_score(y_test_true, y_test_predicted)

    print('')
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")

    false_positives_false_negatives(y_test_true, y_test_predicted, y_probs)


    print(f"ROC Curve (AUC):")
    plot_roc_curve(y_test_true, y_probs)


def plot_roc_curve(y_true, y_probs):
    '''
    This function plots the ROC curve with the AUC metric.
    '''
    fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label = 1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize = (8, 6))
    plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'Curva ROC (Área (AUC) = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc = 'lower right')
    plt.show()   