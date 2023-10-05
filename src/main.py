from utils import load_data, X_y_separation
from dotenv import load_dotenv
import os

if __name__ == '__main__':
    
    #loading of .env variables
    load_dotenv()
    
    path_to_data = os.getenv('PATH_TO_DATA_CSV')
    
    df = load_data(path_to_data)

    X, y = X_y_separation(df, 'stroke')

