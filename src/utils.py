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
