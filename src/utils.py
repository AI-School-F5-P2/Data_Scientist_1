import pandas as pd

from sklearn.model_selection import train_test_split

from collections import Counter

def load_data(path_to_data):
    '''
    This function reads data from a csv file
    and returns a pandas dataframe.
    '''
    try:
        df = pd.read_csv(path_to_data)
        print(df.head())
        return df
    
    except FileNotFoundError:
        print(f"El archivo no se encontró.")
        return None
    
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {str(e)}")
        return None


def X_y_separation(df, string_y):
    '''
    This function receives the dataframe and the name of the 
    target/label column (as a string). It returns the X matrix 
    and the y vector as two different variables.
    '''
    X = df.drop(string_y, axis = 1)
    y = df[string_y].copy()
    return X, y


def split_stratify_y(X, y, test_size):
    '''
    This function receives matrix X and vector y separated
    and the test size. It returns the stratified separation
    (according to y) in train and test. It also verifies if
    the stratification was correctly done.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    count_y = Counter(y)
    count_y_train = Counter(y_train)
    count_y_test = Counter(y_test)

    counters = [count_y, count_y_train, count_y_test]

    for i in counters:
        percentage_of_1 = (i[1]/(i[0]+i[1]))
        print(f'for {i} the percentage of positive instances is: {percentage_of_1}')
    
    return X_train, X_test, y_train, y_test

