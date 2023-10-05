import pandas as pd

from sklearn.model_selection import train_test_split

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