from utils import load_data
from dotenv import load_dotenv
import os

if __name__ == '__main__':
    load_dotenv()
    
    path_to_data = os.getenv('PATH_TO_DATA_CSV')
    
    df = load_data(path_to_data)