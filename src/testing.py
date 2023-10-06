import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc, f1_score
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
    
    probability = model.predict_proba(X_test_transformed)
    y_probs = probability[:, model.classes_.tolist().index(1)]

    return y_test_predicted, y_probs


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

    print(f"Accuracy: {acc}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
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




    